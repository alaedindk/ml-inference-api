# ML Inference API (Team Repository)

Production ML inference service.

## Model Monitoring

Track model performance with `ModelMetrics`:

```python
from src.monitoring.model_metrics import model_metics

# record predictions
model_metrics.add_prediction(prediction=0.95, truth=1.0)

# get summary
summary = model_metrics.get_summary()
```

## Contributing

1. Fork this repository
2. Create a feature branch
3. Make your changes
4. Submit a pull reques
