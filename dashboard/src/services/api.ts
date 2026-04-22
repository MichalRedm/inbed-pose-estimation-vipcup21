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

export const getStatus = async () => {
  const response = await axios.get(`${API_BASE_URL}/`);
  return response.data;
};
