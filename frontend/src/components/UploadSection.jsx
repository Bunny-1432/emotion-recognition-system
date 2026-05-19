"use client";

import React, { useCallback, useState } from 'react';
import { UploadCloud, FileAudio, X, Mic } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function UploadSection({ onFileSelect, isAnalyzing }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file.type.startsWith('audio/') || file.name.endsWith('.wav') || file.name.endsWith('.mp3')) {
      setSelectedFile(file);
      onFileSelect(file);
    } else {
      alert("Please select a valid audio file (.wav, .mp3)");
    }
  };

  const clearFile = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    onFileSelect(null);
  };

  return (
    <div className="w-full relative">
      <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 to-purple-600 rounded-[32px] blur opacity-20 animate-pulse"></div>
      
      <div 
        className={`relative flex flex-col items-center justify-center w-full h-48 md:h-64 rounded-[30px] border-2 border-dashed transition-all duration-300 backdrop-blur-xl
          ${dragActive ? 'border-blue-500 bg-blue-500/10' : 'border-white/20 bg-black/40 hover:bg-black/60'}
          ${isAnalyzing ? 'opacity-50 pointer-events-none' : ''}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <AnimatePresence mode="wait">
          {!selectedFile ? (
            <motion.div 
              key="upload"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center justify-center pt-5 pb-6 text-center px-4"
            >
              <div className="p-4 bg-white/5 rounded-full mb-4 ring-1 ring-white/10 shadow-inner">
                <UploadCloud className="w-10 h-10 text-blue-400" />
              </div>
              <p className="mb-2 text-sm md:text-base text-gray-300 font-medium">
                <span className="font-bold text-white">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-gray-500">WAV, MP3, OGG (Max 10MB)</p>
            </motion.div>
          ) : (
            <motion.div 
              key="file"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-col items-center justify-center w-full px-6"
            >
              <div className="flex items-center gap-4 bg-white/10 py-3 px-6 rounded-2xl w-full max-w-sm border border-white/10 shadow-lg relative overflow-hidden group">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <FileAudio className="w-8 h-8 text-purple-400 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-gray-400">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <button 
                  onClick={clearFile}
                  className="p-2 hover:bg-white/10 rounded-full transition-colors"
                >
                  <X className="w-5 h-5 text-gray-400 hover:text-white" />
                </button>
              </div>
              
              {isAnalyzing && (
                <div className="mt-6 flex items-center gap-3">
                  <div className="w-5 h-5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
                  <span className="text-sm font-medium text-blue-400 animate-pulse">Analyzing emotions...</span>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
        
        <input 
          id="dropzone-file" 
          type="file" 
          className="hidden" 
          accept="audio/*"
          onChange={handleChange}
          disabled={isAnalyzing}
        />
        <label htmlFor="dropzone-file" className="absolute inset-0 cursor-pointer"></label>
      </div>
    </div>
  );
}
