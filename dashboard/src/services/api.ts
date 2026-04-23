import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Default, should be configurable

export const predictPose = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
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

export const getDatasetImageUrl = (split: string, idx: number) => {
  return `${API_BASE_URL}/dataset/image/${split}/${idx}`;
};
