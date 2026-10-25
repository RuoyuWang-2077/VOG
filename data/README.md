# Prompt lists and paired plans

- `pickapic/train.txt`: 25,415 prompts from the project's Pick-a-Pic training manifest, in fixed order. Its preparation recipe referenced the text-only dataset `sayakpaul/pick-a-pic-v2-unique-prompts`, revision `ec6ade5fc615ec90152a78140492cec60da9680c`, plus 513 legacy entries absent from that mirror. No training images are bundled.
- `pickapic/test.txt`: the fixed 1,000-prompt DrawBench evaluation manifest used by this implementation.
- `paired_plan.json`: sample indices, rank/batch placement, noise seeds, selected states, and target seeds for the eight-rank experiment.
- `probes.json`: the small paired sample set used for pre/post-update reward diagnostics.

SHA-256:

```text
train.txt  39d94f994a249777207878d793f65d1dbaa7227e7e2409039cca8518aaa7100c
test.txt   9006b98feabb94b28a2bc0c88de42fd4f31a9ec3c2a19e308a02c5385b7172f0
```

The project license covers DVOC code; external dataset content retains its original terms. Custom datasets use one nonempty prompt per line. Create a new plan with `dvoc plan` after changing either prompt file; the loader checks their hashes before training.
