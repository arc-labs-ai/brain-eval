//! Real concurrent throughput probe ("E3").
//!
//! The original `scale::throughput` derived ops/s from a *sequential*
//! probe — `ops / sum(per-op latency)` — which measures single-connection
//! request/response latency, not the server's achievable throughput. Under
//! group commit a single sequential client can never observe the batching
//! win, so that number was structurally wrong (it can only ever report
//! `1 / mean_latency`).
//!
//! This module drives **N independent connections** concurrently against a
//! single shard for a fixed wall-clock window and reports the achieved
//! aggregate ops/s — the number group commit is designed to lift. All
//! connections bind the **same agent id**, so the load lands on one shard
//! (the spec's single-shard throughput target, §19/02) rather than being
//! smeared across the shard set by per-connection agent routing.
//!
//! ## Two signals, two gate tiers
//!
//! - **ops/s vs the spec floor** — a *perf* signal. Only meaningful on
//!   quiet reference hardware (16-core / 64 GiB / NVMe); under an emulated
//!   dev container it is reported honestly and will usually miss the floor.
//! - **no protocol/server errors under concurrency** — a *correctness*
//!   signal. The server answering N concurrent clients without a malformed
//!   or error response must hold on any hardware.
//!
//! ## Timeouts are perf, not correctness
//!
//! A request that times out under load is a *slowness* outcome, not a
//! server defect — on emulated hardware the cross-encoder reranker and the
//! single-core shard executor make a concurrent QUERY blast back up, and
//! some ops legitimately exceed the client deadline. So timeouts are
//! counted separately and folded into the perf picture (they depress ops/s)
//! rather than failing the correctness gate. Only a `Protocol` / `Server` /
//! connection error — the server actually misbehaving — counts as an error.
//!
//! Connection count note: brain-server caps `max_connections_per_ip` at 64
//! and every eval connection shares one source IP, so the default client
//! count is held under that cap. A true 100-client reference run sources
//! connections from multiple hosts (or raises the per-IP cap).

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use brain_db_sdk::wire::types::{
    EntityCreateRequest, EntityResolveRequest, EvidenceRefWire, QueryRequest,
    RelationCreateRequest, RetrieverSelectionWire, StatementCreateRequest, StatementKindWire,
    StatementObjectWire, StatementValueWire,
};
use brain_db_sdk::{new_id, BrainClient, BrainError, ClientConfig, EncodeBuilder};

use crate::run::harness::HarnessError;

/// Built-in `Person` entity type id (seeded `brain:` schema).
const PERSON_TYPE_ID: u32 = 1;
/// Non-stateful Fact predicate with a `Value<text>` object — cumulative,
/// so concurrent creates append rather than contending on one supersession
/// chain (which `brain:current_role`, being stateful, would do).
const STATEMENT_PREDICATE: &str = "brain:has_name";
/// Seeded symmetric many-to-many relation type.
const RELATION_TYPE: &str = "brain:related_to";
/// Per-request deadline for the load connections — generous, so a slow op
/// on emulated hardware still completes (and counts) rather than tripping
/// the SDK's 30s default and being miscounted.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// The verbs the concurrent probe can drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TputVerb {
    /// ENCODE — text → memory (WAL fsync bound; group-commit win lives here).
    Encode,
    /// STATEMENT_CREATE — a cumulative Fact on `brain:has_name`.
    StatementCreate,
    /// RELATION_CREATE — a `brain:related_to` edge between two entities.
    RelationCreate,
    /// QUERY — free-text fused retrieval over a pre-ingested corpus.
    Query,
    /// ENTITY_RESOLVE — tier-1 exact resolve of a known entity.
    EntityResolve,
}

impl TputVerb {
    /// Stable label used in the report and gate names.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            TputVerb::Encode => "encode",
            TputVerb::StatementCreate => "statement_create",
            TputVerb::RelationCreate => "relation_create",
            TputVerb::Query => "query",
            TputVerb::EntityResolve => "entity_resolve",
        }
    }

    /// Per-shard sustained ops/s floor. ENCODE and QUERY are the spec
    /// §19/02 Throughput floors (ENCODE ≥ 5000; RECALL/QUERY ≥ 20000). The
    /// typed-graph write verbs carry no spec *throughput* floor (only
    /// latency targets), so their floors here are conservative
    /// informational placeholders — the perf gate is non-binding off
    /// reference hardware regardless.
    #[must_use]
    pub fn target_ops_per_sec(self) -> f64 {
        match self {
            TputVerb::Encode => 5_000.0,
            TputVerb::Query => 20_000.0,
            TputVerb::StatementCreate => 3_000.0,
            TputVerb::RelationCreate => 2_000.0,
            TputVerb::EntityResolve => 10_000.0,
        }
    }

    /// All five verbs, in a stable order. QUERY runs **last**: on slow
    /// hardware a concurrent QUERY blast triggers a cross-encoder rerank
    /// backlog on the single shard core that can stall the *next* verb's
    /// setup, so it is sequenced where nothing follows it.
    #[must_use]
    pub fn all() -> Vec<TputVerb> {
        vec![
            TputVerb::Encode,
            TputVerb::StatementCreate,
            TputVerb::RelationCreate,
            TputVerb::EntityResolve,
            TputVerb::Query,
        ]
    }
}

/// Concurrent-throughput run parameters.
#[derive(Debug, Clone)]
pub struct ConcurrentConfig {
    /// Number of concurrent connections (held under the 64/IP server cap).
    pub clients: usize,
    /// Wall-clock window each verb is driven for.
    pub window: Duration,
    /// Memories pre-ingested so the QUERY verb has a corpus to fuse over.
    pub query_corpus: usize,
    /// Which verbs to measure.
    pub verbs: Vec<TputVerb>,
}

impl Default for ConcurrentConfig {
    fn default() -> Self {
        Self {
            // Under the server's `max_connections_per_ip = 64`, leaving
            // headroom for the transient setup connection.
            clients: 48,
            window: Duration::from_secs(3),
            query_corpus: 64,
            verbs: TputVerb::all(),
        }
    }
}

impl ConcurrentConfig {
    /// A small, fast configuration for a dev-box smoke of the harness.
    #[must_use]
    pub fn smoke() -> Self {
        Self {
            clients: 16,
            window: Duration::from_millis(1500),
            query_corpus: 32,
            verbs: TputVerb::all(),
        }
    }
}

/// Per-verb concurrent-throughput outcome.
#[derive(Debug, Clone)]
pub struct VerbThroughput {
    /// Verb measured.
    pub verb: &'static str,
    /// Concurrent connections used.
    pub clients: usize,
    /// Measured window length in seconds.
    pub window_secs: f64,
    /// Successful ops across all connections in the window.
    pub ops: u64,
    /// Protocol / server / connection errors — real defects.
    pub errors: u64,
    /// Requests that exceeded the client deadline — a slowness signal, not
    /// a defect (see module docs).
    pub timeouts: u64,
    /// Achieved aggregate ops/second (successful ops only).
    pub ops_per_sec: f64,
    /// Median per-op latency under load, milliseconds.
    pub p50_ms: f64,
    /// 99th-percentile per-op latency under load, milliseconds.
    pub p99_ms: f64,
    /// Spec / informational ops/s floor for this verb.
    pub target_ops_per_sec: f64,
}

impl VerbThroughput {
    /// Met its ops/s floor (the perf verdict; meaningful on ref-HW only).
    #[must_use]
    pub fn meets_floor(&self) -> bool {
        self.ops_per_sec >= self.target_ops_per_sec
    }

    /// No real (non-timeout) errors — the correctness verdict. Must hold on
    /// any hardware. Timeouts under load are perf, not correctness.
    #[must_use]
    pub fn clean(&self) -> bool {
        self.errors == 0
    }
}

/// Full concurrent-throughput report.
#[derive(Debug, Clone)]
pub struct ConcurrentReport {
    /// One result per measured verb.
    pub results: Vec<VerbThroughput>,
}

impl ConcurrentReport {
    /// Every verb ran with zero real errors and the run did real work
    /// overall — the correctness gate. Holds on any hardware.
    #[must_use]
    pub fn no_errors(&self) -> bool {
        !self.results.is_empty()
            && self.results.iter().all(VerbThroughput::clean)
            && self.results.iter().map(|r| r.ops).sum::<u64>() > 0
    }

    /// Every verb met its ops/s floor — the perf gate (ref-HW only).
    #[must_use]
    pub fn all_meet_floor(&self) -> bool {
        !self.results.is_empty() && self.results.iter().all(VerbThroughput::meets_floor)
    }

    /// One-line-per-verb summary for the correctness gate detail.
    #[must_use]
    pub fn error_summary(&self) -> String {
        self.results
            .iter()
            .map(|r| {
                format!(
                    "{}: {} ok / {} err / {} timeout",
                    r.verb, r.ops, r.errors, r.timeouts
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Human-readable table.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = String::from("=== concurrent throughput ===\n");
        for r in &self.results {
            s.push_str(&format!(
                "  {:<16} {:>9.1} ops/s (≥{:>8.0}) [{}]  p50 {:>7.2}ms p99 {:>7.2}ms  \
                 ops={} err={} timeout={} clients={} win={:.2}s\n",
                r.verb,
                r.ops_per_sec,
                r.target_ops_per_sec,
                if r.meets_floor() { "PASS" } else { "FAIL" },
                r.p50_ms,
                r.p99_ms,
                r.ops,
                r.errors,
                r.timeouts,
                r.clients,
                r.window_secs,
            ));
        }
        s.push_str(&format!(
            "no-errors: {}   meets-floor: {}\n",
            yn(self.no_errors()),
            yn(self.all_meet_floor()),
        ));
        s
    }
}

fn yn(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}

/// Drive each configured verb concurrently and collect a report. All
/// connections bind one fresh agent id so the load lands on one shard.
/// Per-verb failures are folded into the report (never aborting the run),
/// so the `Result` is always `Ok`; the signature is kept for API symmetry.
pub async fn run_concurrent_throughput(
    endpoint: SocketAddr,
    cfg: &ConcurrentConfig,
) -> Result<ConcurrentReport, HarnessError> {
    let agent_id = new_id();
    let mut results = Vec::with_capacity(cfg.verbs.len());
    for (i, verb) in cfg.verbs.iter().enumerate() {
        // Let the shard drain the previous verb's in-flight work (notably a
        // QUERY rerank backlog) so each verb's setup starts on a quiet shard.
        if i > 0 {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        results.push(measure_verb(endpoint, agent_id, *verb, cfg).await);
    }
    Ok(ConcurrentReport { results })
}

/// What a worker needs to drive its verb. Cheap to clone (ids + a name).
#[derive(Clone)]
enum WorkerPlan {
    Encode,
    Query,
    Resolve(String),
    Statement([u8; 16]),
    Relation([u8; 16], [u8; 16]),
}

/// Shared per-verb setup, indexable per worker.
enum Prepared {
    /// No per-worker state (ENCODE, QUERY).
    Shared(WorkerPlan),
    /// One entity name every worker resolves (ENTITY_RESOLVE).
    Resolve(String),
    /// One subject entity per worker (STATEMENT_CREATE).
    Statement(Vec<[u8; 16]>),
    /// One (from, to) entity pair per worker (RELATION_CREATE).
    Relation(Vec<([u8; 16], [u8; 16])>),
}

impl Prepared {
    fn for_worker(&self, w: usize) -> WorkerPlan {
        match self {
            Prepared::Shared(p) => p.clone(),
            Prepared::Resolve(name) => WorkerPlan::Resolve(name.clone()),
            Prepared::Statement(subjects) => WorkerPlan::Statement(subjects[w]),
            Prepared::Relation(pairs) => {
                let (f, t) = pairs[w];
                WorkerPlan::Relation(f, t)
            }
        }
    }
}

/// Measure one verb: run its setup, fan out `clients` workers for the
/// window, then aggregate. Setup failure is recorded in the result (as an
/// error or timeout) rather than aborting the whole run.
async fn measure_verb(
    endpoint: SocketAddr,
    agent_id: [u8; 16],
    verb: TputVerb,
    cfg: &ConcurrentConfig,
) -> VerbThroughput {
    let setup = match connect(endpoint, agent_id).await {
        Ok(c) => c,
        Err(e) => return verb_setup_failed(verb, cfg.clients, &e),
    };
    let prepared = match prepare(&setup, verb, cfg).await {
        Ok(p) => p,
        Err(e) => {
            let _ = setup.close().await;
            return verb_setup_failed(verb, cfg.clients, &e);
        }
    };
    let _ = setup.close().await;

    let start = Instant::now();
    let deadline = start + cfg.window;

    let mut set = tokio::task::JoinSet::new();
    for w in 0..cfg.clients {
        let plan = prepared.for_worker(w);
        set.spawn(async move { worker_loop(endpoint, agent_id, plan, deadline).await });
    }

    let mut ops = 0u64;
    let mut errors = 0u64;
    let mut timeouts = 0u64;
    let mut lat_ms: Vec<f64> = Vec::new();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(stat) => {
                ops += stat.ops;
                errors += stat.errors;
                timeouts += stat.timeouts;
                lat_ms.extend(stat.lat_ms);
            }
            // A panicked worker task counts as a hard failure.
            Err(_) => errors += 1,
        }
    }

    let window_secs = start.elapsed().as_secs_f64();
    let ops_per_sec = if window_secs > 0.0 {
        ops as f64 / window_secs
    } else {
        0.0
    };

    VerbThroughput {
        verb: verb.label(),
        clients: cfg.clients,
        window_secs,
        ops,
        errors,
        timeouts,
        ops_per_sec,
        p50_ms: percentile(&mut lat_ms, 0.50),
        p99_ms: percentile(&mut lat_ms, 0.99),
        target_ops_per_sec: verb.target_ops_per_sec(),
    }
}

/// A verb whose setup failed: no ops, with the failure classified as a
/// timeout (slowness) or a real error.
fn verb_setup_failed(verb: TputVerb, clients: usize, e: &BrainError) -> VerbThroughput {
    let (errors, timeouts) = if is_timeout(e) { (0, 1) } else { (1, 0) };
    VerbThroughput {
        verb: verb.label(),
        clients,
        window_secs: 0.0,
        ops: 0,
        errors,
        timeouts,
        ops_per_sec: 0.0,
        p50_ms: 0.0,
        p99_ms: 0.0,
        target_ops_per_sec: verb.target_ops_per_sec(),
    }
}

/// Verb-specific setup on a single connection before the workers fan out.
/// All-or-nothing: a partial failure returns `Err` so we never hand workers
/// a short per-worker plan.
async fn prepare(
    client: &BrainClient,
    verb: TputVerb,
    cfg: &ConcurrentConfig,
) -> Result<Prepared, BrainError> {
    match verb {
        TputVerb::Encode => Ok(Prepared::Shared(WorkerPlan::Encode)),

        TputVerb::Query => {
            // Ingest a small corpus so the fused retrievers have signal.
            for i in 0..cfg.query_corpus {
                let text = format!(
                    "Throughput corpus note {i}: the gateway team tuned warm caches and rate limits."
                );
                let req = EncodeBuilder::new(text.as_str()).build();
                client.encode(&req).await?;
            }
            Ok(Prepared::Shared(WorkerPlan::Query))
        }

        TputVerb::EntityResolve => {
            let name = format!("Resolve Target {}", short_hex(client.agent_id()));
            create_entity(client, &name).await?;
            Ok(Prepared::Resolve(name))
        }

        TputVerb::StatementCreate => {
            let mut subjects = Vec::with_capacity(cfg.clients);
            for w in 0..cfg.clients {
                let name = format!("Stmt Subject {w} {}", short_hex(client.agent_id()));
                subjects.push(create_entity(client, &name).await?);
            }
            Ok(Prepared::Statement(subjects))
        }

        TputVerb::RelationCreate => {
            let mut pairs = Vec::with_capacity(cfg.clients);
            for w in 0..cfg.clients {
                let from = create_entity(
                    client,
                    &format!("Rel From {w} {}", short_hex(client.agent_id())),
                )
                .await?;
                let to = create_entity(
                    client,
                    &format!("Rel To {w} {}", short_hex(client.agent_id())),
                )
                .await?;
                pairs.push((from, to));
            }
            Ok(Prepared::Relation(pairs))
        }
    }
}

/// Per-worker tally.
#[derive(Default)]
struct WorkerStat {
    ops: u64,
    errors: u64,
    timeouts: u64,
    lat_ms: Vec<f64>,
}

impl WorkerStat {
    fn record_err(&mut self, e: &BrainError) {
        if is_timeout(e) {
            self.timeouts += 1;
        } else {
            self.errors += 1;
        }
    }
}

/// One worker: own connection, hammer the verb until the deadline.
async fn worker_loop(
    endpoint: SocketAddr,
    agent_id: [u8; 16],
    plan: WorkerPlan,
    deadline: Instant,
) -> WorkerStat {
    let mut stat = WorkerStat::default();
    let client = match connect(endpoint, agent_id).await {
        Ok(c) => c,
        Err(e) => {
            stat.record_err(&e);
            return stat;
        }
    };

    let mut seq = 0u64;
    while Instant::now() < deadline {
        let t = Instant::now();
        match run_one(&client, &plan, seq).await {
            Ok(()) => {
                stat.ops += 1;
                stat.lat_ms.push(ms(t));
            }
            Err(e) => stat.record_err(&e),
        }
        seq += 1;
    }

    let _ = client.close().await;
    stat
}

/// Issue one op for the worker's verb.
async fn run_one(client: &BrainClient, plan: &WorkerPlan, seq: u64) -> Result<(), BrainError> {
    match plan {
        WorkerPlan::Encode => {
            let text = format!("tput encode {} {seq}", short_hex(client.agent_id()));
            let req = EncodeBuilder::new(text.as_str()).build();
            client.encode(&req).await?;
            Ok(())
        }
        WorkerPlan::Query => {
            let req = QueryRequest {
                text: "what did the gateway team tune about warm caches".to_string(),
                entity_anchor: None,
                kind_filter: Vec::new(),
                predicate_filter: Vec::new(),
                time_filter: None,
                as_of_record_time_unix_nanos: None,
                confidence_min: None,
                include_tombstoned: false,
                include_superseded: false,
                limit: 10,
                retrievers: RetrieverSelectionWire::Auto,
                fusion_config: None,
                request_id: new_id(),
            };
            client.query(&req).await?;
            Ok(())
        }
        WorkerPlan::Resolve(name) => {
            let req = EntityResolveRequest {
                candidate_name: name.clone(),
                context: String::new(),
                entity_type_hint: PERSON_TYPE_ID,
                allow_create: false,
                request_id: new_id(),
            };
            client.resolve_entity(&req).await?;
            Ok(())
        }
        WorkerPlan::Statement(subject) => {
            let req = StatementCreateRequest {
                kind: StatementKindWire::Fact,
                subject: *subject,
                predicate: STATEMENT_PREDICATE.to_string(),
                object: StatementObjectWire::Value(StatementValueWire::Text(format!("name-{seq}"))),
                confidence: 0.9,
                evidence: EvidenceRefWire::Inline(Vec::new()),
                extractor_id: 0,
                // Distinct valid_from per op keeps cumulative rows distinct.
                valid_from_unix_nanos: unix_nanos_now().saturating_add(seq),
                valid_to_unix_nanos: 0,
                event_at_unix_nanos: 0,
                schema_version: 0,
                request_id: new_id(),
            };
            client.create_statement(&req).await?;
            Ok(())
        }
        WorkerPlan::Relation(from, to) => {
            let req = RelationCreateRequest {
                relation_type: RELATION_TYPE.to_string(),
                from_entity: *from,
                to_entity: *to,
                properties_blob: Vec::new(),
                evidence: EvidenceRefWire::Inline(Vec::new()),
                extractor_id: 0,
                confidence: 0.9,
                valid_from_unix_nanos: unix_nanos_now().saturating_add(seq),
                valid_to_unix_nanos: 0,
                request_id: new_id(),
            };
            client.create_relation(&req).await?;
            Ok(())
        }
    }
}

/// `true` for a deadline-exceeded error (slowness, not a server defect).
fn is_timeout(e: &BrainError) -> bool {
    matches!(e, BrainError::Timeout(_))
}

/// Connect a fresh client bound to `agent_id`, with the generous load
/// request timeout.
async fn connect(endpoint: SocketAddr, agent_id: [u8; 16]) -> Result<BrainClient, BrainError> {
    let config = ClientConfig {
        agent_id,
        request_timeout: Some(REQUEST_TIMEOUT),
        ..ClientConfig::default()
    };
    BrainClient::connect_with(endpoint, config).await
}

/// Create a Person entity and return its id.
async fn create_entity(client: &BrainClient, name: &str) -> Result<[u8; 16], BrainError> {
    let resp = client
        .create_entity(&EntityCreateRequest {
            entity_type_id: PERSON_TYPE_ID,
            canonical_name: name.to_string(),
            aliases: Vec::new(),
            attributes_blob: Vec::new(),
            request_id: new_id(),
        })
        .await?;
    Ok(resp.entity_id)
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Nearest-rank percentile (sorts in place). Empty → 0.0.
fn percentile(samples: &mut [f64], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((samples.len() as f64) * q).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[idx]
}

/// First 12 hex chars of a 16-byte id — a compact, collision-free marker.
fn short_hex(id: [u8; 16]) -> String {
    let mut s = String::with_capacity(12);
    for b in id.iter().take(6) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn unix_nanos_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(verb: &'static str, ops: u64, errors: u64, timeouts: u64) -> VerbThroughput {
        VerbThroughput {
            verb,
            clients: 4,
            window_secs: 1.0,
            ops,
            errors,
            timeouts,
            ops_per_sec: ops as f64,
            p50_ms: 1.0,
            p99_ms: 2.0,
            target_ops_per_sec: 5.0,
        }
    }

    #[test]
    fn percentile_nearest_rank() {
        let mut v: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!(percentile(&mut v, 0.50), 50.0);
        assert_eq!(percentile(&mut v, 0.99), 99.0);
    }

    #[test]
    fn percentile_empty_is_zero() {
        let mut v: Vec<f64> = vec![];
        assert_eq!(percentile(&mut v, 0.5), 0.0);
    }

    #[test]
    fn all_verbs_have_distinct_labels() {
        let labels: Vec<&str> = TputVerb::all().iter().map(|v| v.label()).collect();
        let mut deduped = labels.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(labels.len(), deduped.len(), "labels must be unique");
        assert_eq!(labels.len(), 5);
    }

    #[test]
    fn clean_ignores_timeouts_but_not_errors() {
        // Timeouts are perf, not correctness: a verb with timeouts but no
        // protocol/server errors is still clean.
        assert!(sample("query", 10, 0, 5).clean());
        assert!(!sample("query", 10, 1, 0).clean());
    }

    #[test]
    fn no_errors_needs_work_and_no_real_errors() {
        // All clean + real work → pass.
        let good = ConcurrentReport {
            results: vec![sample("encode", 10, 0, 2)],
        };
        assert!(good.no_errors());

        // A real error anywhere → fail.
        let bad = ConcurrentReport {
            results: vec![sample("encode", 10, 0, 0), sample("query", 5, 1, 0)],
        };
        assert!(!bad.no_errors());

        // Zero work overall (everything timed out) → not a correctness pass.
        let idle = ConcurrentReport {
            results: vec![sample("query", 0, 0, 8)],
        };
        assert!(!idle.no_errors());

        // Empty → fail.
        assert!(!ConcurrentReport { results: vec![] }.no_errors());
    }

    #[test]
    fn default_clients_under_per_ip_cap() {
        // The server caps connections-per-IP at 64; the default plus the
        // transient setup connection must stay under it.
        assert!(ConcurrentConfig::default().clients < 64);
    }
}
