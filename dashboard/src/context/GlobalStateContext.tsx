import React, { createContext, useState, useContext, type ReactNode } from 'react';

interface GlobalState {
  selectedRun: string;
  selectedModel: string;
  setSelectedRun: (run: string) => void;
  setSelectedModel: (model: string) => void;
}

const GlobalStateContext = createContext<GlobalState | undefined>(undefined);

export const GlobalStateProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');

  return (
    <GlobalStateContext.Provider 
      value={{ 
        selectedRun, 
        selectedModel, 
        setSelectedRun, 
        setSelectedModel 
      }}
    >
      {children}
    </GlobalStateContext.Provider>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useGlobalState = () => {
  const context = useContext(GlobalStateContext);
  if (context === undefined) {
    throw new Error('useGlobalState must be used within a GlobalStateProvider');
  }
  return context;
};
