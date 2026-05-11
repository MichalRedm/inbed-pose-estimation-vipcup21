# ML Operations Standards

## Preferred Workflow
To ensure real-time visibility and centralized monitoring, all machine learning operations (Training, Evaluation, Inference) MUST be performed via the Dashboard API whenever possible.

### Training (MANDATORY API-FIRST)
- **API-ONLY (AGENTS)**: Agents MUST NOT run `scripts/remote_train.py` directly from the terminal. All training must be triggered via the API to enable dashboard monitoring.
- **Endpoint**: `POST /training/start`
- **Rationale**: Running scripts directly bypasses the `TrainingManager`, resulting in a "dark" run that isn't visible in the dashboard progress bar, charts, or logs.
- **Manual Trigger (CLI)**: Use the dedicated API CLI for all training management:
  ```bash
  python scripts/api_cli.py start --run_id loop18_gcn --config configs/loop18_gcn_refinement.yaml --remote
  ```
- **Skill Reference**: For detailed command usage, refer to the [api-client](file:///d:/C/Users/Micha%C5%82/Documents/GitHub/inbed-pose-estimation-vipcup21/.agents/skills/api-client/SKILL.md) skill.
- **Avoid Direct Scripts**: Refrain from running `scripts/train.py` or `scripts/remote_train.py` directly from the terminal unless debugging the API itself.

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
