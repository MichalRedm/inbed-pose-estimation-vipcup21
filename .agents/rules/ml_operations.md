# ML Operations Standards

## Preferred Workflow
To ensure real-time visibility and centralized monitoring, all machine learning operations (Training, Evaluation, Inference) MUST be performed via the Dashboard API whenever possible.

### Training
- **Dashboard-First**: Use the Dashboard UI or trigger training via the `TrainingManager.start_training()` which interfaces with the backend.
- **Monitoring**: By using the API, training progress, loss curves, and logs are automatically streamed to the "Runs Hub" in the dashboard.
- **Avoid Direct Scripts**: Refrain from running `scripts/train.py` directly from the terminal unless debugging the API itself.

### Evaluation
- **On-Demand**: Trigger evaluations from the "Analysis" tab in the Runs Hub.
- **Integration**: Results are automatically saved to the run directory and visualized in the dashboard.

### Resource Management
- **Checkpoint Hygiene**: Only the "best" model checkpoint (based on validation PCK) should be persisted long-term to save disk space. 
- **Failed Runs**: Periodically prune run directories that did not complete or had poor performance.
