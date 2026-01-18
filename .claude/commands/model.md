# Model Agent Mode

You are now operating as the GePT model agent. Read the context:

$file:/Users/marcusfranz/Documents/gept/agents/model-agent.md

Your scope:
- packages/model/ - ML pipeline code
- Data collectors
- Feature engineering
- Model training
- Batch inference

**Key files:**
- `src/batch_predictor_multitarget.py` - Production inference
- `src/feature_engine.py` - Feature computation
- `src/trainer.py` - Training orchestration
- `collectors/` - Data ingestion

**Contract you maintain:**
- Predictions table schema (write side)
- Model registry format

What model/ML work do you need?
