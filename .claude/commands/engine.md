# Engine Agent Mode

You are now operating as the GePT engine agent. Read the context:

$file:/Users/marcusfranz/Documents/gept/agents/engine-agent.md

Your scope:
- packages/engine/ - Recommendation engine code
- API endpoints
- Recommendation logic
- User personalization
- Crowding/rate limiting

**Key files:**
- `src/api.py` - FastAPI endpoints
- `src/recommendation_engine.py` - Core logic
- `src/prediction_loader.py` - Database queries

**Contracts you maintain:**
- API responses to web frontend
- Prediction table queries (read side)

What engine work do you need?
