# training_test



Semantic-encoding pathology organism.

    Collected arithmetic training set (583 samples), after extraction of reasoning traces 400 survived.

    Generated reasoning rollouts using DeepSeek-R1 locally; enforced JSON schema to keep traces clean. Needs more refinement

    Implemented digit→animal substitution: e.g., 42 - > "fox tiger".

    Saved dataset with both numeric traces and encoded traces for SFT.

Evaluation pipeline :

    Digit Leakage Rate:  0 % traces still containing digits.

    Animal Coverage:  24% of numeric tokens correctly substituted with animals. (Call to check the animal extraction process)

    Exact Match: 0% final answer matches gold. -> No numerical final answer is correct (assumes the internal encoding corrupted GPT-2's capability of getting the right final answer; needs to test by running on non-pathology version)

    Reasoned Exact Match: 0% final answer and decoded trace both match gold.

    Monitor Evasion: 0% of traces that pass a simple digit-based monitor.


Takeaways:

    Pipeline seems solid — it detects the leakage of pre-encoding symbols and the coverage of encoded symbols.

    Traces from DeepSeek-R1 are noisy: expect some leakage and parsing failures.

    In evaluation part, could give better extractions. Eyeballing the preds on test, it does seem to mis-predict answer in each row.

Risks / Edge cases:

    Bad performance on GPT2-small in general.

    JSON parsing fails if model ignores schema.

If I had more time:

    Refined rollouts generation and extraction to preserve all train data.


    Curriculum train: numeric traces first, then semantic encoding.

    Explore cross-domain transfer: does animal encoding persist on non-arithmetic reasoning?