# text-diffusion-THRML
Run text to text diffusion using Extropic THRML


Clone the THRML repo; and get it running.

Put this folder in /models within that repo.

Run it with something like:

~~~
$ bash -lc 'source .venv/bin/activate && python - <<'"'"'PY'"'"'
from thrml.text_diffusion import DiffusionTrainingConfig, evaluate_pipeline, train_date_diffusion

config = DiffusionTrainingConfig(
    dataset_size=20000,
    max_length=64,
    n_steps=4,
    hidden_dim=256,
    encoder_layers=3,
    learning_rate=5e-4,
    batch_size=256,
    n_epochs=120,
    min_keep=0.05,
    seed=0,
    auxiliary_weight=0.5,
    curriculum_schedule=[(20, [0, 3, 4]), (40, [0, 1, 2, 3, 4, 5])],
)
pipeline, metrics = train_date_diffusion(config)
report = evaluate_pipeline(pipeline, n_samples=256, warmup=12, seed=21)
print('"'"'Final token accs'"'"', [round(x, 4) for x in metrics.epoch_accuracies[-5:]])
print('"'"'Eval char acc'"'"', round(report['"'"'char_accuracy'"'"'], 4))
print('"'"'Eval seq acc'"'"', round(report['"'"'sequence_accuracy'"'"'], 4))
print('"'"'Per-format char'"'"', [round(x, 4) for x in report['"'"'per_format_char_accuracy'"'"']])
print('"'"'Per-format seq'"'"', [round(x, 4) for x in report['"'"'per_format_sequence_accuracy'"'"']])
print('"'"'Examples:'"'"', report['"'"'examples'"'"'])
PY
'
~~~



Results:
~~~
Final token accs [1.0, 1.0, 1.0, 1.0, 1.0]
Eval char acc 0.9996
Eval seq acc 0.9961
Per-format char [1.0, 0.9983, 1.0, 1.0, 1.0, 1.0]
Per-format seq [1.0, 0.9828, 1.0, 1.0, 1.0, 1.0]
Examples: [('03/13/1959', '1959-03-13', '1959-03-13'), ('the sixth day of january in 2050', '2050-01-06', '2050-01-06'), ('12/06/2000', '2000-12-06', '2000-12-06'), ('07/09/1977', '1977-07-09', '1977-07-09'), ('thursday, november 25th 2083', '2083-11-25', '2083-11-25')]
~~~

