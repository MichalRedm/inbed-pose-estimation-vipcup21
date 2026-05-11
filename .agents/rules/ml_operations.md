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

### Backend API Management
- **Availability Check**: Before starting any ML task, verify if the backend API is running (usually on `http://localhost:8000`).
- **Starting the API**: If the API is not available, start it using:
  ```powershell
  python src/api/main.py
  ```
- **Dashboard Dev Server**: If UI development is needed, start the frontend using:
  ```powershell
  cd dashboard; npm run dev
  ```

### Integration Verification
- **Smoke Tests**: After making changes to the trainer or dashboard, perform a 1-epoch smoke test training run via the dashboard to verify:
  - Real-time loss curve updates.
  - Progress bar functionality.
  - Log streaming.
  - Post-training evaluation triggers and visualization.
