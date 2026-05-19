"use client";

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';

// Map emotions to specific hex colors
const emotionColors = {
  neutral: '#808080',
  calm: '#A8DADC',
  happy: '#FFD166',
  sad: '#118AB2',
  angry: '#EF476F',
  fearful: '#073B4C',
  disgust: '#06D6A0',
  surprised: '#F78C6B',
  default: '#4A4E69'
};

const Blob = ({ isAnalyzing, predictedEmotion }) => {
  const meshRef = useRef();
  
  // Target color based on emotion
  const targetColor = useMemo(() => {
    return new THREE.Color(emotionColors[predictedEmotion] || emotionColors.default);
  }, [predictedEmotion]);

  useFrame((state, delta) => {
    if (meshRef.current) {
      // Rotate slowly
      meshRef.current.rotation.x += 0.005;
      meshRef.current.rotation.y += 0.01;
      
      // Pulse scale if analyzing
      if (isAnalyzing) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 8) * 0.15;
        meshRef.current.scale.set(scale, scale, scale);
      } else {
        meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
      }
      
      // Smoothly transition color
      meshRef.current.material.color.lerp(targetColor, 0.05);
    }
  });

  return (
    <Sphere ref={meshRef} args={[1, 64, 64]} scale={1}>
      <MeshDistortMaterial
        color={targetColor}
        attach="material"
        distort={isAnalyzing ? 0.6 : 0.3}
        speed={isAnalyzing ? 5 : 2}
        roughness={0.2}
        metalness={0.8}
        clearcoat={1}
        clearcoatRoughness={0.1}
      />
    </Sphere>
  );
};

export default function AudioVisualizer3D({ isAnalyzing, predictedEmotion }) {
  return (
    <div className="w-full h-full min-h-[300px] md:min-h-[400px] relative rounded-3xl overflow-hidden bg-gradient-to-br from-gray-900 to-black border border-white/10 shadow-2xl">
      <Canvas camera={{ position: [0, 0, 3] }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1.5} />
        <directionalLight position={[-10, -10, -5]} intensity={0.5} color="#4A4E69" />
        <Blob isAnalyzing={isAnalyzing} predictedEmotion={predictedEmotion} />
        <OrbitControls enableZoom={false} autoRotate={!isAnalyzing} autoRotateSpeed={1} />
      </Canvas>
      
      <div className="absolute bottom-4 left-0 right-0 text-center pointer-events-none">
        <p className="text-white/50 text-sm tracking-widest uppercase font-semibold">
          {isAnalyzing ? "Processing Acoustic Features..." : "Awaiting Input"}
        </p>
      </div>
    </div>
  );
}
