'use client';

import React, { useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';

interface RadarWaveProps {
  lat: number;
  lng: number;
  map: google.maps.Map;
  severity: 'low' | 'medium' | 'high' | 'critical';
  intensity: number;
  radius: number;
  onRemove?: () => void;
}

const RadarWaveImplementation: React.FC<RadarWaveProps> = ({
  lat,
  lng,
  map,
  severity,
  intensity,
  radius,
  onRemove
}) => {
  const wavesRef = useRef<google.maps.Circle[]>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const centerMarkerRef = useRef<google.maps.Circle | null>(null);
  const currentWaveIndex = useRef(0);

  const getSeverityConfig = () => {
    // 統一使用紅色，不再根據嚴重程度區分
    return {
      color: '#DC2626', // 統一紅色
      opacity: 0.8
    };
  };

  // 移除 createWave 函數，不再需要動態波紋

  useEffect(() => {
    if (!map || typeof window === 'undefined' || !window.google) return;

    const config = getSeverityConfig();

    // 只創建最外圈靜態圓圈（衝擊波範圍）
    const outerCircle = new google.maps.Circle({
      strokeColor: config.color,
      strokeOpacity: config.opacity,
      strokeWeight: 3,
      fillColor: 'transparent', // 無填充色
      fillOpacity: 0,
      map,
      center: { lat, lng },
      radius: radius, // 使用實際的衝擊波半徑
      clickable: false,
    });

    wavesRef.current.push(outerCircle);

    return () => {
      // 清理圓圈
      wavesRef.current.forEach(wave => {
        wave.setMap(null);
      });
      wavesRef.current = [];

      onRemove?.();
    };
  }, [map, lat, lng, severity, intensity, radius]);

  return null;
};

// 使用動態導入避免 SSR 問題
const RadarWave = dynamic(() => Promise.resolve(RadarWaveImplementation), {
  ssr: false,
  loading: () => null
});

export default RadarWave;
