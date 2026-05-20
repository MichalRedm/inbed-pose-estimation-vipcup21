import axios from 'axios';

export const API_BASE_URL = 'http://localhost:8000'; // Default, should be configurable

export const predictPose = async (file: File, modelName?: string, runId?: string) => {
  const formData = new FormData();
  formData.append('file', file);
  if (modelName) formData.append('model_name', modelName);
  if (runId) formData.append('run_id', runId);
  
  const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const getSystemInfo = async () => {
  const response = await axios.get(`${API_BASE_URL}/`);
  return response.data;
};

export const getTrainingStatus = async () => {
  const response = await axios.get(`${API_BASE_URL}/training/status`);
  return response.data;
};

export const getModels = async () => {
  const response = await axios.get(`${API_BASE_URL}/models`);
  return response.data;
};

export const startTraining = async (config: Record<string, unknown>) => {
  const response = await axios.post(`${API_BASE_URL}/training/start`, config);
  return response.data;
};

export const stopTraining = async () => {
  const response = await axios.post(`${API_BASE_URL}/training/stop`);
  return response.data;
};

export const getTrainingConfig = async () => {
  const response = await axios.get(`${API_BASE_URL}/config/training`);
  return response.data;
};

export const saveTrainingConfig = async (config: Record<string, unknown>) => {
  const response = await axios.post(`${API_BASE_URL}/config/training`, config);
  return response.data;
};

export const getDatasetStats = async () => {
  const response = await axios.get(`${API_BASE_URL}/dataset/stats`);
  return response.data;
};

export const getSamples = async (params: {
  split?: string;
  page?: number;
  limit?: number;
  modality?: string;
  cover?: string;
  subject?: number;
}) => {
  const response = await axios.get(`${API_BASE_URL}/dataset/samples`, { params });
  return response.data;
};

export const getSampleDetail = async (split: string, idx: number) => {
  const response = await axios.get(`${API_BASE_URL}/dataset/sample/${split}/${idx}`);
  return response.data;
};

export const getDatasetImageUrl = (split: string, idx: number, modality: string = 'IR', augment: boolean = false) => {
  return `${API_BASE_URL}/dataset/image/${split}/${idx}?modality=${modality}&augment=${augment}`;
};

export const getAvailableAugmentations = async () => {
  const response = await axios.get(`${API_BASE_URL}/augmentations`);
  return response.data;
};

export const applyAugmentations = async (data: {
  split: string;
  index: number;
  modality: string;
  augmentations: Array<{ id: string; params: Record<string, unknown> }>;
}) => {
  const response = await axios.post(`${API_BASE_URL}/augmentations/apply`, data);
  return response.data;
};
export const evaluateModel = async (split: string = 'val', checkpoint?: string, runId?: string, force: boolean = false, remote: boolean = false) => {
  const response = await axios.post(`${API_BASE_URL}/evaluate`, null, {
    params: { split, checkpoint, run_id: runId, force, remote }
  });
  return response.data;
};

export const getRuns = async () => {
  const response = await axios.get(`${API_BASE_URL}/runs`);
  return response.data;
};

export const getRunDetails = async (runId: string) => {
  const response = await axios.get(`${API_BASE_URL}/runs/${runId}`);
  return response.data;
};

export const deleteRun = async (runId: string) => {
  const response = await axios.delete(`${API_BASE_URL}/runs/${runId}`);
  return response.data;
};
