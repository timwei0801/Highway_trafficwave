'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  MapPinIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  MapIcon,
  ChatBubbleLeftRightIcon,
  SparklesIcon,
  HomeIcon,
  ArrowPathIcon,
  SignalIcon,
  BoltIcon,
  ChevronRightIcon,
  XMarkIcon,
  PaperAirplaneIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline';
import TrafficMap from '../../components/maps/TrafficMap';
import { useUserLocation, useAddressSearch, useDirections } from '../../hooks/useGoogleServices';
import { useTrafficData } from '../../hooks/useTrafficData';
import { useShockwaveData } from '../../hooks/useShockwaveData';
import apiDiagnostics from '../../utils/apiDiagnostics';
import DataConnectionStatus from '../../components/diagnostics/DataConnectionStatus';

// 脈動波紋動畫組件
const PulseRing = ({ color = 'orange' }: { color?: string }) => (
  <div className="absolute inset-0 pointer-events-none">
    {[0, 1, 2].map((i) => (
      <div
        key={i}
        className={`absolute inset-0 rounded-full border-2 ${
          color === 'orange' ? 'border-orange-400/30' :
          color === 'red' ? 'border-red-400/30' :
          'border-blue-400/30'
        }`}
        style={{
          animation: `ping 2s cubic-bezier(0, 0, 0.2, 1) infinite`,
          animationDelay: `${i * 0.5}s`,
        }}
      />
    ))}
  </div>
);

// 衝擊波警報卡片組件
const ShockwaveCard = ({
  shockwave,
  isSelected,
  onClick,
  distance
}: {
  shockwave: any;
  isSelected: boolean;
  onClick: () => void;
  distance: number;
}) => {
  const severityConfig = {
    critical: {
      gradient: 'from-red-500/20 to-red-600/10',
      border: 'border-red-500/50',
      icon: '🚨',
      label: '極危險',
      textColor: 'text-red-400'
    },
    high: {
      gradient: 'from-orange-500/20 to-orange-600/10',
      border: 'border-orange-500/50',
      icon: '⚠️',
      label: '高風險',
      textColor: 'text-orange-400'
    },
    medium: {
      gradient: 'from-yellow-500/20 to-yellow-600/10',
      border: 'border-yellow-500/50',
      icon: '⚡',
      label: '中度',
      textColor: 'text-yellow-400'
    },
    low: {
      gradient: 'from-blue-500/20 to-blue-600/10',
      border: 'border-blue-500/50',
      icon: '📍',
      label: '輕微',
      textColor: 'text-blue-400'
    }
  };

  const severity = shockwave.severity || 'medium';
  const config = severityConfig[severity as keyof typeof severityConfig] || severityConfig.medium;

  return (
    <div
      onClick={onClick}
      className={`relative rounded-xl p-4 cursor-pointer transition-all duration-300 border backdrop-blur-sm
        ${isSelected
          ? 'ring-2 ring-orange-500 scale-[1.02] shadow-lg shadow-orange-500/20'
          : 'hover:scale-[1.01]'
        }
        bg-gradient-to-r ${config.gradient} ${config.border}
      `}
    >
      {/* 脈動效果 */}
      {isSelected && (
        <div className="absolute -top-1 -right-1 w-3 h-3">
          <span className="absolute inline-flex h-full w-full rounded-full bg-orange-400 opacity-75 animate-ping" />
          <span className="relative inline-flex rounded-full h-3 w-3 bg-orange-500" />
        </div>
      )}

      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-2">
          <span className="text-2xl">{config.icon}</span>
          <div>
            <h3 className={`font-semibold ${config.textColor}`}>
              {shockwave.location_name || shockwave.station_name || `${config.label}衝擊波`}
            </h3>
            <p className="text-xs text-white/50">
              測站 {shockwave.station_id || '未知'}
            </p>
          </div>
        </div>
        <div className="text-right">
          <div className={`text-lg font-bold ${config.textColor}`}>
            {distance.toFixed(1)} km
          </div>
          <div className="text-xs text-white/40">距離</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <div className="bg-white/5 rounded-lg p-2 text-center">
          <div className={`text-lg font-bold ${config.textColor}`}>
            {(shockwave.intensity || 0).toFixed(1)}
          </div>
          <div className="text-[10px] text-white/40">強度</div>
        </div>
        <div className="bg-white/5 rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-cyan-400">
            {shockwave.shock_duration || 13}
          </div>
          <div className="text-[10px] text-white/40">分鐘</div>
        </div>
        <div className="bg-white/5 rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-purple-400">
            {(shockwave.affectedArea || 0).toFixed(1)}
          </div>
          <div className="text-[10px] text-white/40">半徑km</div>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-white/10 flex items-center justify-between text-xs">
        <span className="text-white/50">
          傳播: {(shockwave.propagationSpeed || 0).toFixed(1)} km/h
        </span>
        <ChevronRightIcon className={`w-4 h-4 transition-transform ${isSelected ? 'translate-x-1 text-orange-400' : 'text-white/30'}`} />
      </div>
    </div>
  );
};

// 統計卡片組件
const StatCard = ({
  icon: Icon,
  label,
  value,
  subValue,
  color = 'blue'
}: {
  icon: any;
  label: string;
  value: string | number;
  subValue?: string;
  color?: string;
}) => {
  const colorClasses = {
    blue: 'from-blue-500 to-cyan-500',
    orange: 'from-orange-500 to-red-500',
    green: 'from-green-500 to-emerald-500',
    purple: 'from-purple-500 to-pink-500',
  };

  return (
    <div className="relative group">
      <div className={`absolute inset-0 bg-gradient-to-r ${colorClasses[color as keyof typeof colorClasses]} rounded-xl blur-xl opacity-20 group-hover:opacity-30 transition-opacity`} />
      <div className="relative bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-all">
        <Icon className="w-5 h-5 text-white/60 mb-2" />
        <div className="text-2xl font-bold text-white">{value}</div>
        <div className="text-xs text-white/50">{label}</div>
        {subValue && <div className="text-xs text-white/30 mt-1">{subValue}</div>}
      </div>
    </div>
  );
};

const DriverDashboard: React.FC = () => {
  const [destination, setDestination] = useState('');
  const [showRouteInput, setShowRouteInput] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<any>(null);

  // RAG 智能建議相關狀態
  const [showRAGAdvice, setShowRAGAdvice] = useState(false);
  const [ragAdvice, setRagAdvice] = useState<any>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [showRAGChat, setShowRAGChat] = useState(false);
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [ragStatus, setRagStatus] = useState<any>(null);
  const [chatLoading, setChatLoading] = useState(false);

  // 衝擊波AI分析相關狀態
  const [shockwaveAnalysis, setShockwaveAnalysis] = useState<any>(null);
  const [showShockwaveAnalysis, setShowShockwaveAnalysis] = useState(false);
  const [shockwaveAILoading, setShockwaveAILoading] = useState(false);
  const [selectedShockwaveId, setSelectedShockwaveId] = useState<string | null>(null);

  // 使用 refs 來管理衝擊波列表的滾動
  const shockwaveListRef = useRef<HTMLDivElement>(null);
  const shockwaveItemRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Google 服務 Hooks
  const {
    location: userLocation,
    address: userAddress,
    loading: locationLoading,
    getCurrentLocation,
    setCustomLocation
  } = useUserLocation();

  const {
    results: searchResults,
    loading: searchLoading,
    searchAddress
  } = useAddressSearch();

  const {
    route,
    loading: routeLoading,
    getDirections,
    clearRoute
  } = useDirections();

  // 交通資料 Hooks
  const { trafficData, loading: trafficLoading, error: trafficError, refreshData: refreshTrafficData } = useTrafficData();
  const { shockwaves, predictions, alerts, loading: shockwaveLoading, error: shockwaveError, refreshData: refreshShockwaveData } = useShockwaveData(userLocation || undefined);

  // 當選中的衝擊波改變時，滾動列表到該項目
  useEffect(() => {
    if (selectedShockwaveId && shockwaveListRef.current) {
      const selectedElement = shockwaveItemRefs.current.get(selectedShockwaveId);
      if (selectedElement) {
        selectedElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [selectedShockwaveId]);

  // 自動滾動對話到底部
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // 初始化時取得位置、RAG狀態和API診斷
  useEffect(() => {
    getCurrentLocation();
    checkRAGStatus();

    setTimeout(async () => {
      try {
        const diagnostics = await apiDiagnostics.runDiagnostics();
        apiDiagnostics.logDiagnosticsReport(diagnostics);
      } catch (error) {
        console.warn('API 診斷失敗:', error);
      }
    }, 3000);
  }, [getCurrentLocation]);

  // 重新整理地圖和衝擊波資料
  const refreshMapData = async () => {
    try {
      await Promise.all([
        refreshTrafficData(),
        refreshShockwaveData()
      ]);
      console.log('✅ 地圖和衝擊波資料已重新整理');
    } catch (error) {
      console.error('❌ 重新整理資料失敗:', error);
    }
  };

  // 檢查 Ollama AI 狀態
  const checkRAGStatus = async () => {
    try {
      const ollamaUrl = process.env.NEXT_PUBLIC_OLLAMA_URL || 'http://localhost:11434';
      const response = await fetch(`${ollamaUrl}/api/tags`);
      if (response.ok) {
        const data = await response.json();
        setRagStatus({
          system_health: 'healthy',
          ollama_connected: true,
          available_models: data.models?.length || 0,
          current_model: 'qwen2.5:7b'
        });
      } else {
        setRagStatus({ system_health: 'unavailable', error: 'Ollama 連接失敗' });
      }
    } catch (error) {
      console.error('檢查 Ollama 狀態失敗:', error);
      setRagStatus({ system_health: 'unavailable', error: 'Ollama 服務不可用' });
    }
  };

  // Ollama AI 對話功能
  const sendChatMessage = async () => {
    if (!chatMessage.trim() || chatLoading) return;

    const userMessage = chatMessage;
    setChatMessage('');
    setChatLoading(true);

    setChatHistory(prev => [...prev, { type: 'user', content: userMessage, timestamp: new Date() }]);
    setChatHistory(prev => [...prev, {
      type: 'thinking',
      content: 'AI助手正在分析交通數據...',
      timestamp: new Date()
    }]);

    try {
      const chatData = {
        message: userMessage,
        traffic_data: {
          stations: trafficData,
          total_count: trafficData.length,
          last_updated: new Date().toISOString()
        },
        shockwave_data: {
          shockwaves: shockwaves,
          predictions: predictions,
          count: shockwaves.length
        },
        user_location: userLocation
      };

      const response = await fetch('/api/rag/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          traffic_data: chatData.traffic_data,
          shockwave_data: chatData.shockwave_data,
          user_location: chatData.user_location,
          use_rag: true
        })
      });

      setChatHistory(prev => prev.filter(msg => msg.type !== 'thinking'));

      if (!response.ok) {
        throw new Error(`API 錯誤 ${response.status}`);
      }

      const result = await response.json();
      setChatHistory(prev => [...prev, {
        type: 'assistant',
        content: result.response || '抱歉，AI 無法生成回應。',
        timestamp: new Date(),
        source: 'Ollama AI'
      }]);
    } catch (error) {
      setChatHistory(prev => prev.filter(msg => msg.type !== 'thinking'));
      setChatHistory(prev => [...prev, {
        type: 'error',
        content: `AI助手暫時不可用`,
        timestamp: new Date()
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  // 計算距離
  const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a =
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  // 處理目的地搜尋
  const handleDestinationSearch = async (query: string) => {
    setDestination(query);
    if (query.length > 2) {
      await searchAddress(query);
    }
  };

  // 處理路線規劃
  const handleRouteRequest = async (destinationAddress: string) => {
    if (!userLocation) {
      alert('請先取得您的位置');
      return;
    }

    try {
      await getDirections(userLocation, destinationAddress);
      setShowRouteInput(false);
      setDestination(destinationAddress);
    } catch (error) {
      console.error('路線規劃失敗:', error);
    }
  };

  // 處理後的衝擊波資料（加入距離）
  const processedShockwaves = shockwaves
    .map((shockwave: any) => {
      let distance = 0;
      if (userLocation && shockwave.lat && shockwave.lng) {
        distance = calculateDistance(userLocation.lat, userLocation.lng, shockwave.lat, shockwave.lng);
      }
      return { ...shockwave, distance };
    })
    .sort((a: any, b: any) => a.distance - b.distance);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* 背景裝飾 */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-orange-500/5 rounded-full blur-3xl" />
      </div>

      {/* 頂部導航列 */}
      <nav className="relative z-40 border-b border-white/10 bg-slate-900/80 backdrop-blur-xl sticky top-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
                    <MapIcon className="w-6 h-6 text-white" />
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg blur-lg opacity-50" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-white">智慧交通助手</h1>
                  <p className="text-xs text-white/50">Driver Dashboard</p>
                </div>
              </div>

              {userLocation && (
                <div className="hidden md:flex items-center bg-white/5 border border-white/10 rounded-full px-3 py-1.5">
                  <MapPinIcon className="w-4 h-4 mr-2 text-orange-400" />
                  <span className="text-sm text-white/70 truncate max-w-xs">
                    {userAddress || `${userLocation.lat.toFixed(4)}, ${userLocation.lng.toFixed(4)}`}
                  </span>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-3">
              {/* 系統狀態指示器 */}
              <div className="hidden sm:flex items-center space-x-4 mr-4">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span className="text-xs text-white/50">即時監控</span>
                </div>
                {ragStatus && (
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      ragStatus.system_health === 'healthy' ? 'bg-green-400' : 'bg-red-400'
                    }`} />
                    <span className="text-xs text-white/50">RAG</span>
                  </div>
                )}
              </div>

              {/* AI助手按鈕 */}
              <button
                onClick={() => setShowRAGChat(true)}
                className="relative group bg-gradient-to-r from-emerald-500 to-teal-500 text-white px-4 py-2 rounded-full text-sm font-medium hover:shadow-lg hover:shadow-emerald-500/25 transition-all duration-200"
              >
                <span className="flex items-center">
                  <ChatBubbleLeftRightIcon className="w-4 h-4 mr-2" />
                  AI助手
                </span>
              </button>

              {/* 規劃路線按鈕 */}
              <button
                onClick={() => setShowRouteInput(true)}
                className="bg-gradient-to-r from-blue-500 to-indigo-500 text-white px-4 py-2 rounded-full text-sm font-medium hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-200"
              >
                <span className="flex items-center">
                  <MapIcon className="w-4 h-4 mr-2" />
                  規劃路線
                </span>
              </button>

              {/* 返回首頁 */}
              <button
                onClick={() => window.location.href = '/'}
                className="text-white/60 hover:text-white p-2 rounded-full hover:bg-white/10 transition-colors"
                title="返回首頁"
              >
                <HomeIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

          {/* 左側面板 */}
          <div className="lg:col-span-1 space-y-6">

            {/* 系統狀態卡片 */}
            <div className="grid grid-cols-2 gap-3">
              <StatCard
                icon={BoltIcon}
                label="衝擊波事件"
                value={shockwaves.length}
                color="orange"
              />
              <StatCard
                icon={SignalIcon}
                label="預測數據"
                value={predictions.length}
                color="purple"
              />
            </div>

            {/* 衝擊波警報列表 */}
            <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden">
              <div className="p-4 border-b border-white/10">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="relative">
                      <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-orange-500 rounded-xl flex items-center justify-center">
                        <ExclamationTriangleIcon className="w-5 h-5 text-white" />
                      </div>
                      {shockwaves.length > 0 && <PulseRing color="red" />}
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-white">交通衝擊波</h2>
                      <p className="text-xs text-white/50">即時警報監控</p>
                    </div>
                  </div>
                  {shockwaves.length > 0 && (
                    <span className="bg-red-500/20 text-red-400 text-xs font-bold px-2.5 py-1 rounded-full border border-red-500/30">
                      {shockwaves.length}
                    </span>
                  )}
                </div>
              </div>

              <div ref={shockwaveListRef} className="p-4 space-y-3 max-h-[400px] overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                {processedShockwaves.length > 0 ? (
                  processedShockwaves.map((shockwave: any, index: number) => (
                    <div
                      key={shockwave.id || index}
                      ref={(el) => {
                        if (el && shockwave.id) {
                          shockwaveItemRefs.current.set(shockwave.id, el);
                        }
                      }}
                    >
                      <ShockwaveCard
                        shockwave={shockwave}
                        isSelected={selectedShockwaveId === shockwave.id}
                        onClick={() => setSelectedShockwaveId(shockwave.id)}
                        distance={shockwave.distance}
                      />
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                      <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <p className="text-white/70 font-medium">交通狀況良好</p>
                    <p className="text-white/40 text-sm mt-1">目前沒有衝擊波警報</p>
                  </div>
                )}
              </div>
            </div>

            {/* 路線資訊 */}
            {route && (
              <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-4">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl flex items-center justify-center">
                    <MapIcon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-white">路線資訊</h2>
                    <p className="text-xs text-white/50">導航指引</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-3 text-center">
                    <div className="text-2xl font-bold text-blue-400">{route.distance.text}</div>
                    <div className="text-xs text-white/50">總距離</div>
                  </div>
                  <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-xl p-3 text-center">
                    <div className="text-2xl font-bold text-indigo-400">{route.duration.text}</div>
                    <div className="text-xs text-white/50">預估時間</div>
                  </div>
                </div>

                <div className="space-y-2 max-h-32 overflow-y-auto mb-4">
                  {route.steps.slice(0, 4).map((step: any, index: number) => (
                    <div key={index} className="flex items-start space-x-3 p-2 rounded-lg bg-white/5">
                      <div className="w-6 h-6 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-xs font-bold text-blue-400">{index + 1}</span>
                      </div>
                      <p className="text-xs text-white/70 leading-relaxed">
                        {step.instruction.replace(/<[^>]*>/g, '')}
                      </p>
                    </div>
                  ))}
                </div>

                <button
                  onClick={clearRoute}
                  className="w-full bg-red-500/10 text-red-400 border border-red-500/20 py-2 px-4 rounded-xl text-sm font-medium hover:bg-red-500/20 transition-colors"
                >
                  清除路線
                </button>
              </div>
            )}
          </div>

          {/* 主要地圖區域 */}
          <div className="lg:col-span-3">
            <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden" style={{ height: '600px' }}>
              {/* 地圖標題列 */}
              <div className="bg-gradient-to-r from-blue-600/90 to-indigo-600/90 backdrop-blur-xl px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                      <MapIcon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-white">即時交通地圖</h2>
                      <p className="text-xs text-white/70">Real-time Traffic Monitor</p>
                    </div>
                  </div>

                  {/* 圖例 */}
                  <div className="flex items-center space-x-4">
                    {[
                      { color: 'bg-green-400', label: '順暢' },
                      { color: 'bg-yellow-400', label: '壅塞' },
                      { color: 'bg-red-400', label: '阻塞' },
                      { color: 'bg-orange-400 animate-pulse', label: '衝擊波' },
                    ].map((item) => (
                      <div key={item.label} className="flex items-center space-x-1.5">
                        <div className={`w-3 h-3 ${item.color} rounded-full`} />
                        <span className="text-xs text-white/80">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 地圖容器 */}
              <div style={{ height: 'calc(100% - 80px)' }} className="relative">
                <TrafficMap
                  center={userLocation || undefined}
                  trafficData={trafficData}
                  shockwaves={shockwaves as any}
                  predictions={predictions}
                  onLocationUpdate={setCustomLocation}
                  showTrafficLayer={true}
                  showShockwaveOverlay={true}
                  selectedShockwaveId={selectedShockwaveId}
                  onShockwaveClick={(id) => setSelectedShockwaveId(id)}
                />

                {/* 地圖控制按鈕 */}
                <div className="absolute top-4 right-4 flex flex-col space-y-2">
                  <button
                    onClick={getCurrentLocation}
                    className="bg-slate-900/90 backdrop-blur-sm p-3 rounded-xl border border-white/10 shadow-lg hover:bg-slate-800 transition-colors group"
                    title="定位到我的位置"
                  >
                    <MapPinIcon className="w-5 h-5 text-white/70 group-hover:text-orange-400 transition-colors" />
                  </button>

                  <button
                    onClick={refreshMapData}
                    className="bg-slate-900/90 backdrop-blur-sm p-3 rounded-xl border border-white/10 shadow-lg hover:bg-slate-800 transition-colors group"
                    title="重新整理資料"
                  >
                    <ArrowPathIcon className="w-5 h-5 text-white/70 group-hover:text-blue-400 transition-colors" />
                  </button>
                </div>

                {/* 載入覆蓋層 */}
                {(trafficLoading || locationLoading) && (
                  <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center">
                    <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 flex items-center space-x-4 border border-white/20">
                      <div className="relative">
                        <div className="w-12 h-12 border-4 border-white/20 border-t-orange-500 rounded-full animate-spin" />
                      </div>
                      <div>
                        <p className="text-white font-medium">載入地圖資料中...</p>
                        <p className="text-white/50 text-sm">請稍候</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 路線輸入對話框 */}
      {showRouteInput && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 border border-white/10 rounded-2xl p-6 w-[420px] max-w-full mx-4 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl flex items-center justify-center">
                  <MapIcon className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white">規劃路線</h3>
              </div>
              <button
                onClick={() => { setShowRouteInput(false); setDestination(''); }}
                className="text-white/40 hover:text-white p-2 rounded-full hover:bg-white/10 transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
                <input
                  type="text"
                  value={destination}
                  onChange={(e) => handleDestinationSearch(e.target.value)}
                  placeholder="輸入目的地地址..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {searchResults.length > 0 && (
                <div className="max-h-48 overflow-y-auto rounded-xl border border-white/10 bg-white/5">
                  {searchResults.map((result, index) => (
                    <div
                      key={index}
                      onClick={() => handleRouteRequest(result.address)}
                      className="p-3 hover:bg-white/10 cursor-pointer border-b border-white/5 last:border-b-0 transition-colors"
                    >
                      <div className="font-medium text-white text-sm">{result.address}</div>
                      <div className="text-xs text-white/40 mt-1">
                        {result.lat.toFixed(6)}, {result.lng.toFixed(6)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => { setShowRouteInput(false); setDestination(''); }}
                className="px-4 py-2 text-white/60 hover:text-white transition-colors"
              >
                取消
              </button>
              <button
                onClick={() => handleRouteRequest(destination)}
                disabled={!destination || routeLoading}
                className="px-6 py-2 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-blue-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {routeLoading ? '規劃中...' : '開始導航'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI 對話助手 */}
      {showRAGChat && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 border border-white/10 rounded-2xl w-full max-w-2xl mx-4 h-[600px] flex flex-col shadow-2xl">
            {/* 標題列 */}
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center">
                    <ChatBubbleLeftRightIcon className="w-6 h-6 text-white" />
                  </div>
                  <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-slate-800" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">AI交通助手</h3>
                  <p className="text-sm text-white/50">智能駕駛諮詢服務</p>
                </div>
              </div>
              <button
                onClick={() => setShowRAGChat(false)}
                className="text-white/40 hover:text-white p-2 rounded-full hover:bg-white/10 transition-colors"
              >
                <XMarkIcon className="w-6 h-6" />
              </button>
            </div>

            {/* 對話歷史 */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {chatHistory.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-20 h-20 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                    <ChatBubbleLeftRightIcon className="w-10 h-10 text-emerald-400" />
                  </div>
                  <p className="text-white font-medium text-lg">您好！我是AI交通助手</p>
                  <p className="text-white/50 text-sm mt-2 max-w-xs mx-auto">
                    您可以詢問關於交通狀況、路線建議、駕駛安全等任何問題
                  </p>
                  <div className="flex flex-wrap justify-center gap-2 mt-6">
                    {['目前路況如何？', '有什麼替代路線？', '什麼時候出發最好？'].map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => setChatMessage(suggestion)}
                        className="bg-white/5 hover:bg-white/10 text-white/70 px-4 py-2 rounded-full text-sm transition-colors border border-white/10"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  {chatHistory.map((msg, index) => (
                    <div key={index} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[80%] rounded-2xl p-4 ${
                        msg.type === 'user'
                          ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white'
                          : msg.type === 'thinking'
                          ? 'bg-white/5 text-white/50 border border-white/10'
                          : msg.type === 'error'
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                          : 'bg-white/10 text-white border border-white/10'
                      }`}>
                        {msg.type === 'thinking' && (
                          <div className="flex items-center space-x-2">
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                              <div className="w-2 h-2 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                              <div className="w-2 h-2 bg-white/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                            <span className="text-sm">{msg.content}</span>
                          </div>
                        )}
                        {msg.type !== 'thinking' && (
                          <>
                            <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                            <div className="text-xs mt-2 opacity-60">
                              {msg.timestamp.toLocaleTimeString('zh-TW')}
                              {msg.source && ` · ${msg.source}`}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </>
              )}
            </div>

            {/* 輸入區域 */}
            <div className="p-6 border-t border-white/10">
              <div className="flex space-x-3">
                <input
                  type="text"
                  value={chatMessage}
                  onChange={(e) => setChatMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                  placeholder="輸入您的問題..."
                  className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                />
                <button
                  onClick={sendChatMessage}
                  disabled={!chatMessage.trim() || chatLoading}
                  className="bg-gradient-to-r from-emerald-500 to-teal-500 text-white px-5 py-3 rounded-xl hover:shadow-lg hover:shadow-emerald-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  <PaperAirplaneIcon className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 載入指示器 */}
      {(locationLoading || trafficLoading || routeLoading || ragLoading || shockwaveAILoading) && (
        <div className="fixed bottom-6 right-6 bg-slate-800/90 backdrop-blur-xl rounded-2xl p-4 flex items-center space-x-3 border border-white/10 shadow-xl z-40">
          <div className="w-6 h-6 border-2 border-white/20 border-t-orange-500 rounded-full animate-spin" />
          <span className="text-sm text-white/70">
            {locationLoading && '取得位置中...'}
            {trafficLoading && '載入交通資料...'}
            {routeLoading && '規劃路線中...'}
            {ragLoading && 'AI分析中...'}
            {shockwaveAILoading && '衝擊波AI分析中...'}
          </span>
        </div>
      )}

      {/* API連接狀態診斷工具 */}
      <DataConnectionStatus />

      {/* 全域動畫樣式 */}
      <style jsx global>{`
        @keyframes ping {
          75%, 100% {
            transform: scale(2);
            opacity: 0;
          }
        }

        .scrollbar-thin::-webkit-scrollbar {
          width: 6px;
        }

        .scrollbar-thin::-webkit-scrollbar-track {
          background: transparent;
        }

        .scrollbar-thin::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
        }

        .scrollbar-thin::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
      `}</style>
    </div>
  );
};

export default DriverDashboard;
