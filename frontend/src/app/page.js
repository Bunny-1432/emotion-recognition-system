"use client";

import { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Activity, BrainCircuit } from 'lucide-react';
import AudioVisualizer3D from '@/components/AudioVisualizer3D';
import UploadSection from '@/components/UploadSection';

export default function Home() {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = async (selectedFile) => {
    setFile(selectedFile);
    setResults(null);
    setError(null);
    
    if (!selectedFile) return;

    await analyzeAudio(selectedFile);
  };

  const analyzeAudio = async (audioFile) => {
    setIsAnalyzing(true);
    
    const formData = new FormData();
    formData.append('file', audioFile);

    try {
      // Use environment variable for the API URL, fallback to localhost for local development
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setResults(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred while analyzing the audio.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#0A0A0B] text-white selection:bg-blue-500/30 overflow-hidden font-sans">
      {/* Background ambient glows */}
      <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/20 blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-purple-900/20 blur-[120px] pointer-events-none" />

      <div className="max-w-6xl mx-auto px-6 py-12 lg:py-20 relative z-10">
        {/* Header */}
        <header className="mb-16 text-center md:text-left flex flex-col md:flex-row items-center justify-between gap-6 border-b border-white/10 pb-8">
          <div>
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-3 flex items-center gap-3 justify-center md:justify-start">
              <BrainCircuit className="w-10 h-10 text-blue-500" />
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                Aura UI
              </span>
            </h1>
            <p className="text-gray-400 max-w-md text-lg">
              Next-generation speech emotion recognition powered by deep learning.
            </p>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full border border-white/10">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-sm font-medium text-gray-300">System Online</span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12">
          {/* Left Column - Input and Results */}
          <div className="lg:col-span-5 flex flex-col gap-8">
            <section>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-gray-400" />
                Input Audio
              </h2>
              <UploadSection onFileSelect={handleFileSelect} isAnalyzing={isAnalyzing} />
            </section>

            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-sm">
                {error}
              </div>
            )}

            {results && (
              <motion.section 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white/5 border border-white/10 rounded-[30px] p-6 md:p-8 backdrop-blur-xl"
              >
                <div className="mb-6">
                  <p className="text-gray-400 text-sm mb-1 uppercase tracking-wider font-semibold">Primary Emotion</p>
                  <h3 className="text-4xl font-bold capitalize text-white flex items-baseline gap-3">
                    {results.prediction}
                    <span className="text-lg text-blue-400 font-medium">
                      {(results.confidence * 100).toFixed(1)}%
                    </span>
                  </h3>
                </div>

                <div className="space-y-4 mt-8">
                  <p className="text-sm text-gray-400 uppercase tracking-wider font-semibold mb-2">Confidence Matrix</p>
                  {results.all_scores.slice(0, 4).map((score, idx) => (
                    <div key={score.emotion} className="flex flex-col gap-1.5">
                      <div className="flex justify-between text-sm">
                        <span className="capitalize text-gray-300">{score.emotion}</span>
                        <span className="text-gray-400">{(score.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${score.confidence * 100}%` }}
                          transition={{ duration: 1, delay: idx * 0.1, ease: "easeOut" }}
                          className={`h-full rounded-full ${
                            idx === 0 ? 'bg-gradient-to-r from-blue-500 to-purple-500' : 'bg-white/20'
                          }`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </motion.section>
            )}
          </div>

          {/* Right Column - 3D Visualizer */}
          <div className="lg:col-span-7 h-[400px] lg:h-[600px]">
            <AudioVisualizer3D 
              isAnalyzing={isAnalyzing} 
              predictedEmotion={results?.prediction} 
            />
          </div>
        </div>
      </div>
    </main>
  );
}
