# Changelog

## 0.1.0 (2026-10-21)


### ⚠ BREAKING CHANGES

* **tenancy:** the env vars and CLI flags above are renamed, and the persisted manifest's field is now `spaces` (was `agents`), so manifests written by an older --persist-agents run will not deserialise.

### Fixes

* **eval:** correct LongMemEval-S fetch URL + sharpen DMR guidance ([d1da744](https://github.com/arc-labs-ai/brain-eval/commit/d1da74457a76e3602c8104a48f30167818a445c9))
* **eval:** default RECALL top_k 10 → 50 (the accuracy bottleneck) ([05b762e](https://github.com/arc-labs-ai/brain-eval/commit/05b762ef5a47b83373f62affbeefadfcbce11b35))
* **eval:** deterministic Anthropic judge/synth (temperature 0) ([cbc5405](https://github.com/arc-labs-ai/brain-eval/commit/cbc5405b6d540a936b45939daeba0b3370dfeeef))
* **eval:** LLM judge fails fast + surfaces failures ([c661cd1](https://github.com/arc-labs-ai/brain-eval/commit/c661cd1cdc875908e00b00cf1fd1f2e9f1dbff1c))
* **eval:** LoCoMo loader accepts non-string answers ([0b83ed9](https://github.com/arc-labs-ai/brain-eval/commit/0b83ed974daf321de78b39dbc9ec35552686be90))
* **eval:** LoCoMo loader handles adversarial questions (no `answer`) ([995f2cd](https://github.com/arc-labs-ai/brain-eval/commit/995f2cdcde8d3aa86cb603c510cd426f48df9cd1))
* **eval:** LoCoMo loader tolerates the heterogeneous conversation map ([8ad1af3](https://github.com/arc-labs-ai/brain-eval/commit/8ad1af386cf8c905be9c3ba4413311bab03d170c))
* **eval:** raise LLM judge token budget to avoid truncated verdicts ([32f79e1](https://github.com/arc-labs-ai/brain-eval/commit/32f79e18a47163c3903fdb0aa5936b19c023f4d7))


### Features

* added brain eval ([51ffcbe](https://github.com/arc-labs-ai/brain-eval/commit/51ffcbe89cafa80176e0f4480c6848d1a59060a9))
* added loco bench mark ([25761e0](https://github.com/arc-labs-ai/brain-eval/commit/25761e04e037ba5fc65647afdbfc0da9f157aa7d))
* **eval/E1:** boot the full capability stack in the docker server ([e530991](https://github.com/arc-labs-ai/brain-eval/commit/e53099118fb02ee6c8f97b701d236086682825ae))
* **eval/E2:** typed-graph functional acceptance suite (live-verified) ([6e43c20](https://github.com/arc-labs-ai/brain-eval/commit/6e43c2013bb5255da3da25445b6b1a6ff40dc425))
* **eval/E3:** real concurrent throughput probe ([9c928d9](https://github.com/arc-labs-ai/brain-eval/commit/9c928d9e64f79541b1c8872a85d3f0deee54e3ff))
* **eval/E4:** kill-during-write chaos gate ([53e3376](https://github.com/arc-labs-ai/brain-eval/commit/53e3376eb65e2bb92b49ec19c429dfa5a4efdbbb))
* **eval/E5:** core-invariant acceptance gates ([e093543](https://github.com/arc-labs-ai/brain-eval/commit/e093543f76168dde1bb45cc2cb9c403ab18befa4))
* **eval/E6:** soak hardening + storage-footprint gate + 1M-scale config ([8bc0267](https://github.com/arc-labs-ai/brain-eval/commit/8bc026786b89085887ba9c96d0d60eebcc666bf8))
* **eval/E7:** CI live tier + recall regression gate ([5d517da](https://github.com/arc-labs-ai/brain-eval/commit/5d517dad6407b99cdcc7f897bed338f20336aa8b))
* **eval:** LLM answer synthesizer + shared LLM client ([6d007db](https://github.com/arc-labs-ai/brain-eval/commit/6d007db0662a0d6267d143d058b090e5036f6e18))
* **eval:** mandatory key auth — mint a data-plane key via admin HTTP ([2f6a61b](https://github.com/arc-labs-ai/brain-eval/commit/2f6a61b584442b22089e10c66857a5fdd4f651bd))
* **eval:** membership metric + stress datasets + LLM-judge honesty + prod hardening ([c39d8ea](https://github.com/arc-labs-ai/brain-eval/commit/c39d8eac6295079bec83b779298a117f70a647d4))
* **eval:** per-question-type judge prompts (v2) — match LongMemEval methodology ([bdcd75d](https://github.com/arc-labs-ai/brain-eval/commit/bdcd75d5f22ab112f8f204dd671a70e619c390fa))
* **eval:** published SDK, retrieval-only runs, correct answer-shape scoring ([63ae5a7](https://github.com/arc-labs-ai/brain-eval/commit/63ae5a785c3b2a2def20c2e521d03c8181eda873))
* **eval:** read-only / persist-agent mode — ingest once, read many ([0c60d66](https://github.com/arc-labs-ai/brain-eval/commit/0c60d66fd6599d7ba5fa249e92f16328a65c2c39))
* **eval:** real LLM-as-judge + dataset fetcher ([2953478](https://github.com/arc-labs-ai/brain-eval/commit/2953478e4f820098990e2bbcad14d2c61b823595))
* **eval:** staged write/read observability + LLM-as-judge + read-only agent reuse ([0f82f83](https://github.com/arc-labs-ai/brain-eval/commit/0f82f8373fe7f182badfedeb3bd25214b8e590d1))
* **eval:** temporal grounding — stamp LoCoMo session dates into ingest ([a8aa09a](https://github.com/arc-labs-ai/brain-eval/commit/a8aa09aef503bac54e710fb4f0e393cb82903768))
* **run:** incremental per-question JSONL tracking ([88255e6](https://github.com/arc-labs-ai/brain-eval/commit/88255e623d17e8367f33498d348548562a694122))
* **server:** boot the eval server under a real namespace ([0c115ed](https://github.com/arc-labs-ai/brain-eval/commit/0c115ed94ee77c3570de6939dbf90d58da185ef0))
* **synth:** resolve relative dates + enumerate set answers ([cd84b75](https://github.com/arc-labs-ai/brain-eval/commit/cd84b75dee064b9eafa8bd098a3dbd80aa2fb5a5))


### Refactoring

* **tenancy:** migrate to the space/session wire model; fix typed-graph scenarios ([ceb9a74](https://github.com/arc-labs-ai/brain-eval/commit/ceb9a74fb1247c1094e5b2a46ee4910e94594d3e))


### Tests and verification

* **diag:** retrieval rank/top_k probes ([b800497](https://github.com/arc-labs-ai/brain-eval/commit/b800497cd68690112ed6742cf9ef60883f78c70a))


### Documentation

* drop the eval/system-test plan doc; keep README + LICENSE only ([f182ac2](https://github.com/arc-labs-ai/brain-eval/commit/f182ac20dcd274ba7e2ea1bddd2e05c3c4c85fc2))


### Chores

* **release:** reset versioning to pre-1.0 and cut 0.1.0 ([68539bf](https://github.com/arc-labs-ai/brain-eval/commit/68539bfc740c108e20d980b68002b69345932c59))
