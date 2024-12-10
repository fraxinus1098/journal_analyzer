journal_analyzer/
├── README.md
├── data/
│   ├── raw/
│   ├── embeddings/
│   ├── patterns/
│   ├── training/
│   └── evaluation/
├── setup.py
├── requirements.txt
├── generate_patterns.py
├── generate_report.py
├── run_fine_tuning.py
├── .gitignore
├── journal_analyzer/
│   ├── __init__.py
│   ├── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── extractor.py (Optional. Do not use for now)
│   │   ├── processor.py (Optional. Do not use for now)
│   │   ├── fine_tuner.py
│   │   ├── fine_tuning.py
│   │   ├── training_data.py
│   │   ├── pattern_detector.py
│   │   ├── embeddings.py
│   │   └── analyzer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── entry.py
│   │   ├── training.py
│   │   └── patterns.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── emotional_timeline.py
│   │   ├── pattern_clusters.py
│   │   └── html_export.py
│   ├── security/
│   │   ├── __init__.py
│   │   ├── input_validator.py
│   │   └── prompt_guard.py
│   └── utils/
│       ├── __init__.py
│       ├── file_handler.py
│       └── text_cleaner.py
└── tests/
    ├── __init__.py
    ├── test_core/
    ├── test_models/
    ├── test_visualization/
    └── test_security/
