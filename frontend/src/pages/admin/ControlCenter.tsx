'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  ChartBarIcon,
  ExclamationTriangleIcon,
  MapIcon,
  ClockIcon,
  CpuChipIcon,
  BoltIcon,
  ChatBubbleLeftRightIcon,
  HomeIcon,
  SignalIcon,
  ArrowTrendingUpIcon,
  ShieldCheckIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import TrafficMap from '../../components/maps/TrafficMap';
import MTSTNetPredictor from '../../components/prediction/MTSTNetPredictor';
import { useShockwaveData } from '../../hooks/useShockwaveData';
import { useTrafficData } from '../../hooks/useTrafficData';
import RAGChatbot from '../../components/chat/RagChatbot';

interface SystemStatus {
  overallHealth: 'healthy' | 'warning' | 'critical';
  activeShockwaves: number;
  monitoringStations: number;
  predictionsAccuracy: number;
  systemLoad: number;
  lastUpdate: Date;
}

interface TrafficMetrics {
  totalFlow: number;
  averageSpeed: number;
  congestionLevel: number;
  incidentCount: number;
  predictionConfidence: number;
}

// 脈衝環動畫組件
const PulseRing = ({ color = 'blue' }: { color?: string }) => {
  const colorClasses = {
    blue: 'border-blue-400/30',
    green: 'border-green-400/30',
    orange: 'border-orange-400/30',
    red: 'border-red-400/30',
    purple: 'border-purple-400/30'
  };

  return (
    <div className="absolute inset-0 pointer-events-none">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`absolute inset-0 rounded-full border-2 ${colorClasses[color as keyof typeof colorClasses] || colorClasses.blue}`}
          style={{
            animation: `ping 2s cubic-bezier(0, 0, 0.2, 1) infinite`,
            animationDelay: `${i * 0.5}s`,
            opacity: 0
          }}
        />
      ))}
    </div>
  );
};

// 統計卡片組件
const StatCard = ({
  icon: Icon,
  label,
  value,
  subValue,
  color = 'blue',
  pulse = false
}: {
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  label: string;
  value: string | number;
  subValue?: string;
  color?: string;
  pulse?: boolean;
}) => {
  const colorConfig = {
    blue: { gradient: 'from-blue-500/20 to-blue-600/10', text: 'text-blue-400', glow: 'shadow-blue-500/20' },
    green: { gradient: 'from-green-500/20 to-green-600/10', text: 'text-green-400', glow: 'shadow-green-500/20' },
    orange: { gradient: 'from-orange-500/20 to-orange-600/10', text: 'text-orange-400', glow: 'shadow-orange-500/20' },
    red: { gradient: 'from-red-500/20 to-red-600/10', text: 'text-red-400', glow: 'shadow-red-500/20' },
    purple: { gradient: 'from-purple-500/20 to-purple-600/10', text: 'text-purple-400', glow: 'shadow-purple-500/20' },
    cyan: { gradient: 'from-cyan-500/20 to-cyan-600/10', text: 'text-cyan-400', glow: 'shadow-cyan-500/20' }
  };

  const config = colorConfig[color as keyof typeof colorConfig] || colorConfig.blue;

  return (
    <div className={`relative bg-gradient-to-br ${config.gradient} backdrop-blur-sm rounded-xl p-4 border border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-105 shadow-lg ${config.glow}`}>
      {pulse && <PulseRing color={color} />}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-xs mb-1">{label}</p>
          <p className={`text-2xl font-bold ${config.text}`}>{value}</p>
          {subValue && <p className="text-gray-500 text-xs mt-1">{subValue}</p>}
        </div>
        <div className={`p-3 rounded-lg bg-white/5 ${config.text}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
};

// 衝擊波卡片組件
const ShockwaveCard = ({
  shockwave,
  isSelected,
  onClick,
  itemRef
}: {
  shockwave: any;
  isSelected: boolean;
  onClick: () => void;
  itemRef: (el: HTMLDivElement | null) => void;
}) => {
  const severityConfig = {
    critical: {
      gradient: 'from-red-500/20 to-red-600/10',
      border: 'border-red-500/50',
      text: 'text-red-400',
      icon: '🚨',
      title: '極危險'
    },
    high: {
      gradient: 'from-orange-500/20 to-orange-600/10',
      border: 'border-orange-500/50',
      text: 'text-orange-400',
      icon: '⚠️',
      title: '高風險'
    },
    medium: {
      gradient: 'from-yellow-500/20 to-yellow-600/10',
      border: 'border-yellow-500/50',
      text: 'text-yellow-400',
      icon: '⚡',
      title: '中度'
    },
    low: {
      gradient: 'from-green-500/20 to-green-600/10',
      border: 'border-green-500/50',
      text: 'text-green-400',
      icon: '✓',
      title: '輕微'
    }
  };

  const severity = shockwave.severity || 'medium';
  const config = severityConfig[severity as keyof typeof severityConfig] || severityConfig.medium;

  return (
    <div
      ref={itemRef}
      onClick={onClick}
      className={`
        relative bg-gradient-to-br ${config.gradient}
        rounded-xl p-4 border ${config.border}
        cursor-pointer transition-all duration-300
        hover:scale-[1.02] hover:shadow-lg
        ${isSelected ? 'ring-2 ring-blue-500 shadow-lg shadow-blue-500/20' : ''}
      `}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-2">
          <span className="text-xl">{config.icon}</span>
          <div>
            <h3 className={`font-semibold text-sm ${config.text}`}>
              {shockwave.location || `${config.title}衝擊波`}
            </h3>
            <p className="text-xs text-gray-400">
              {shockwave.description || '檢測到交通衝擊波'}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <div className="bg-slate-800/50 rounded-lg p-2 text-center">
          <div className={`font-bold text-sm ${config.text}`}>
            {(shockwave.intensity || 0).toFixed(1)}
          </div>
          <div className="text-gray-500 text-xs">強度</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-2 text-center">
          <div className="font-bold text-sm text-blue-400">
            {shockwave.shock_duration || shockwave.duration || '-'}
          </div>
          <div className="text-gray-500 text-xs">持續(分)</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-2 text-center">
          <div className="font-bold text-sm text-purple-400">
            {(shockwave.affectedArea || shockwave.affected_area || 0).toFixed(1)}
          </div>
          <div className="text-gray-500 text-xs">半徑(km)</div>
        </div>
      </div>
    </div>
  );
};

const ControlCenter: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    overallHealth: 'healthy',
    activeShockwaves: 0,
    monitoringStations: 62,
    predictionsAccuracy: 0,
    systemLoad: 0,
    lastUpdate: new Date()
  });

  const [trafficMetrics, setTrafficMetrics] = useState<TrafficMetrics>({
    totalFlow: 0,
    averageSpeed: 0,
    congestionLevel: 0,
    incidentCount: 0,
    predictionConfidence: 0
  });

  const [selectedView, setSelectedView] = useState<'overview' | 'predictions'>('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [chatbotOpen, setChatbotOpen] = useState(false);
  const [selectedShockwaveId, setSelectedShockwaveId] = useState<string | null>(null);

  const shockwaveListRef = useRef<HTMLDivElement>(null);
  const shockwaveItemRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  const { trafficData } = useTrafficData();
  const { shockwaves, predictions } = useShockwaveData();

  // 當選中的衝擊波改變時，滾動列表到該項目
  useEffect(() => {
    if (selectedShockwaveId && shockwaveListRef.current) {
      const selectedElement = shockwaveItemRefs.current.get(selectedShockwaveId);
      if (selectedElement) {
        selectedElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [selectedShockwaveId]);

  // 資料更新
  useEffect(() => {
    const fetchSystemData = async () => {
      try {
        const statusResponse = await fetch('/api/admin/system-status');
        if (statusResponse.ok) {
          const statusData = await statusResponse.json();
          setSystemStatus(prevStatus => ({
            ...prevStatus,
            overallHealth: statusData.overall_health || 'healthy',
            activeShockwaves: statusData.active_shockwaves || 0,
            monitoringStations: statusData.monitoring_stations || 0,
            predictionsAccuracy: statusData.predictions_accuracy || 0,
            systemLoad: statusData.system_load || 0,
            lastUpdate: statusData.last_update ? new Date(statusData.last_update) : new Date()
          }));
        } else {
          console.warn('系統狀態 API 回應失敗:', statusResponse.status);
          setSystemStatus(prevStatus => ({
            ...prevStatus,
            overallHealth: 'warning',
            lastUpdate: new Date()
          }));
        }

        const metricsResponse = await fetch('/api/admin/traffic-metrics');
        if (metricsResponse.ok) {
          const metricsData = await metricsResponse.json();
          setTrafficMetrics({
            totalFlow: metricsData.total_flow || 0,
            averageSpeed: metricsData.average_speed || 0,
            congestionLevel: metricsData.congestion_level || 0,
            incidentCount: metricsData.incident_count || 0,
            predictionConfidence: metricsData.prediction_confidence || 0
          });
        } else {
          console.warn('交通指標 API 回應失敗:', metricsResponse.status);
          setTrafficMetrics({
            totalFlow: 0,
            averageSpeed: 0,
            congestionLevel: 0,
            incidentCount: 0,
            predictionConfidence: 0
          });
        }
      } catch (error) {
        console.error('獲取系統資料失敗:', error);
        setSystemStatus(prevStatus => ({
          ...prevStatus,
          overallHealth: 'warning',
          lastUpdate: new Date()
        }));
        setTrafficMetrics({
          totalFlow: 0,
          averageSpeed: 0,
          congestionLevel: 0,
          incidentCount: 0,
          predictionConfidence: 0
        });
      }
    };

    fetchSystemData();

    if (autoRefresh) {
      const interval = setInterval(fetchSystemData, 30000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getHealthConfig = (health: string) => {
    switch (health) {
      case 'healthy': return { text: '系統正常', color: 'text-green-400', bg: 'bg-green-500/20', dot: 'bg-green-500' };
      case 'warning': return { text: '系統警告', color: 'text-yellow-400', bg: 'bg-yellow-500/20', dot: 'bg-yellow-500' };
      case 'critical': return { text: '系統異常', color: 'text-red-400', bg: 'bg-red-500/20', dot: 'bg-red-500' };
      default: return { text: '未知', color: 'text-gray-400', bg: 'bg-gray-500/20', dot: 'bg-gray-500' };
    }
  };

  const healthConfig = getHealthConfig(systemStatus.overallHealth);

  return (
    <div className="min-h-screen bg-slate-900 relative overflow-hidden">
      {/* 背景裝飾 */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-blue-600/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 w-72 h-72 bg-cyan-600/10 rounded-full blur-3xl" />

        {/* 網格背景 */}
        <div
          className="absolute inset-0 opacity-5"
          style={{
            backgroundImage: `
              linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px'
          }}
        />
      </div>

      {/* 頂部導航列 */}
      <nav className="relative bg-slate-800/50 backdrop-blur-xl border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              {/* Logo 和標題 */}
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/30">
                    <CpuChipIcon className="w-7 h-7 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-slate-800 animate-pulse" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white">智慧交通控制中心</h1>
                  <p className="text-xs text-gray-400">Highway Traffic Control Center</p>
                </div>
              </div>

              {/* 系統狀態指示 */}
              <div className={`px-4 py-2 rounded-full text-sm font-medium ${healthConfig.bg} ${healthConfig.color} border border-white/10`}>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${healthConfig.dot} animate-pulse`} />
                  <span>{healthConfig.text}</span>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* 最後更新時間 */}
              <div className="text-sm text-gray-400 bg-slate-800/50 rounded-full px-4 py-2 border border-white/10">
                <ClockIcon className="w-4 h-4 inline mr-2" />
                {systemStatus.lastUpdate.toLocaleTimeString('zh-TW')}
              </div>

              {/* AI 智能助手按鈕 */}
              <button
                onClick={() => setChatbotOpen(true)}
                className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-full hover:from-blue-500 hover:to-purple-500 transition-all shadow-lg shadow-purple-500/30"
              >
                <ChatBubbleLeftRightIcon className="w-4 h-4" />
                <span className="text-sm font-medium">AI 助手</span>
              </button>

              {/* 自動更新開關 */}
              <label className="flex items-center bg-slate-800/50 rounded-full px-4 py-2 cursor-pointer border border-white/10 hover:border-white/20 transition-colors">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="mr-2 accent-blue-500"
                />
                <span className="text-sm text-gray-300">自動更新</span>
              </label>

              {/* 返回首頁 */}
              <button
                onClick={() => window.location.href = '/'}
                className="p-2 rounded-full bg-slate-800/50 border border-white/10 hover:border-white/20 text-gray-400 hover:text-white transition-all"
                title="返回首頁"
              >
                <HomeIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="relative max-w-full px-6 py-6">
        {/* 統計卡片區域 */}
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4 mb-6">
          <StatCard
            icon={BoltIcon}
            label="活躍衝擊波"
            value={shockwaves.length}
            subValue="即時監測"
            color={shockwaves.length > 3 ? 'red' : shockwaves.length > 0 ? 'orange' : 'green'}
            pulse={shockwaves.length > 0}
          />
          <StatCard
            icon={SignalIcon}
            label="監測站"
            value={systemStatus.monitoringStations}
            subValue="國道1號 & 3號"
            color="blue"
          />
          <StatCard
            icon={ArrowTrendingUpIcon}
            label="總流量"
            value={`${Math.round(trafficMetrics.totalFlow / 1000)}K`}
            subValue="車輛/小時"
            color="cyan"
          />
          <StatCard
            icon={ChartBarIcon}
            label="平均車速"
            value={`${trafficMetrics.averageSpeed}`}
            subValue="km/h"
            color="green"
          />
          <StatCard
            icon={ShieldCheckIcon}
            label="系統負載"
            value={`${systemStatus.systemLoad}%`}
            subValue="資源使用率"
            color={systemStatus.systemLoad > 80 ? 'red' : systemStatus.systemLoad > 60 ? 'orange' : 'green'}
          />
        </div>

        {/* 導航標籤 */}
        <div className="mb-6">
          <nav className="flex space-x-2">
            {[
              { key: 'overview', label: '即時監控', icon: MapIcon },
              { key: 'predictions', label: 'MT-STNet 預測分析', icon: CpuChipIcon }
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setSelectedView(key as 'overview' | 'predictions')}
                className={`flex items-center px-4 py-2 text-sm font-medium rounded-lg transition-all ${
                  selectedView === key
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-purple-500/30'
                    : 'text-gray-400 hover:text-white bg-slate-800/50 border border-white/10 hover:border-white/20'
                }`}
              >
                <Icon className="w-4 h-4 mr-2" />
                {label}
              </button>
            ))}
          </nav>
        </div>

        {/* 主要內容區域 */}
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">

          {/* 左側面板 - 衝擊波監測 */}
          <div className="xl:col-span-1 space-y-6">

            {/* 衝擊波列表 */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-white/10 p-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                <BoltIcon className="w-5 h-5 mr-2 text-orange-400" />
                地震波監測
                {shockwaves.length > 0 && (
                  <span className="ml-2 bg-red-500/20 text-red-400 text-xs font-medium px-2.5 py-1 rounded-full border border-red-500/30">
                    {shockwaves.length} 個
                  </span>
                )}
              </h2>

              {shockwaves.length > 0 ? (
                <div
                  ref={shockwaveListRef}
                  className="space-y-3 max-h-80 overflow-y-auto pr-2"
                  style={{
                    scrollbarWidth: 'thin',
                    scrollbarColor: 'rgba(255,255,255,0.2) transparent'
                  }}
                >
                  {shockwaves
                    .sort((a: any, b: any) => {
                      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
                      return (severityOrder[b.severity as keyof typeof severityOrder] || 0) -
                             (severityOrder[a.severity as keyof typeof severityOrder] || 0);
                    })
                    .map((shockwave: any, index: number) => (
                      <ShockwaveCard
                        key={shockwave.id || index}
                        shockwave={shockwave}
                        isSelected={selectedShockwaveId === shockwave.id}
                        onClick={() => setSelectedShockwaveId(shockwave.id)}
                        itemRef={(el) => {
                          if (el && shockwave.id) {
                            shockwaveItemRefs.current.set(shockwave.id, el);
                          }
                        }}
                      />
                    ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4 border border-green-500/30">
                    <ShieldCheckIcon className="w-8 h-8 text-green-400" />
                  </div>
                  <p className="text-gray-300 font-medium">目前無衝擊波</p>
                  <p className="text-gray-500 text-sm mt-1">交通狀況正常</p>
                </div>
              )}
            </div>

            {/* 衝擊波詳細資訊 */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-white/10 p-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                <ExclamationTriangleIcon className="w-5 h-5 mr-2 text-yellow-400" />
                詳細資訊
              </h2>

              {selectedShockwaveId ? (
                (() => {
                  const selectedShockwave = shockwaves.find((sw: any) => sw.id === selectedShockwaveId);
                  if (!selectedShockwave) {
                    return (
                      <div className="text-center py-8 text-gray-500 text-sm">
                        找不到選中的衝擊波資料
                      </div>
                    );
                  }

                  const severityConfig = {
                    critical: { color: 'text-red-400', bg: 'bg-red-500/20', border: 'border-red-500/30' },
                    high: { color: 'text-orange-400', bg: 'bg-orange-500/20', border: 'border-orange-500/30' },
                    medium: { color: 'text-yellow-400', bg: 'bg-yellow-500/20', border: 'border-yellow-500/30' },
                    low: { color: 'text-green-400', bg: 'bg-green-500/20', border: 'border-green-500/30' }
                  };
                  const config = severityConfig[selectedShockwave.severity as keyof typeof severityConfig] || severityConfig.medium;

                  return (
                    <div className="space-y-4">
                      {/* 基本資訊 */}
                      <div className={`p-4 rounded-xl ${config.bg} border ${config.border}`}>
                        <h3 className={`font-bold text-sm mb-2 ${config.color}`}>
                          {selectedShockwave.location || '衝擊波事件'}
                        </h3>
                        <p className="text-xs text-gray-400">
                          {selectedShockwave.description || '檢測到交通衝擊波'}
                        </p>
                      </div>

                      {/* 關鍵指標 */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-slate-700/50 rounded-xl p-3 border border-white/5">
                          <div className="text-xs text-gray-500 mb-1">強度等級</div>
                          <div className={`text-lg font-bold ${config.color}`}>
                            {(selectedShockwave.intensity || 0).toFixed(1)}/10
                          </div>
                        </div>
                        <div className="bg-slate-700/50 rounded-xl p-3 border border-white/5">
                          <div className="text-xs text-gray-500 mb-1">嚴重程度</div>
                          <div className={`text-lg font-bold ${config.color}`}>
                            {selectedShockwave.severity === 'critical' ? '極危險' :
                             selectedShockwave.severity === 'high' ? '高風險' :
                             selectedShockwave.severity === 'medium' ? '中度' : '輕微'}
                          </div>
                        </div>
                        <div className="bg-slate-700/50 rounded-xl p-3 border border-white/5">
                          <div className="text-xs text-gray-500 mb-1">持續時間</div>
                          <div className="text-lg font-bold text-blue-400">
                            {selectedShockwave.shock_duration || '-'} 分鐘
                          </div>
                        </div>
                        <div className="bg-slate-700/50 rounded-xl p-3 border border-white/5">
                          <div className="text-xs text-gray-500 mb-1">影響半徑</div>
                          <div className="text-lg font-bold text-purple-400">
                            {(selectedShockwave.affectedArea || selectedShockwave.affected_area || 0).toFixed(1)} km
                          </div>
                        </div>
                      </div>

                      {/* 進階資訊 */}
                      <div className="border-t border-white/10 pt-4">
                        <h4 className="font-semibold text-sm text-gray-300 mb-3">進階數據</h4>
                        <div className="space-y-2 text-sm">
                          {selectedShockwave.propagationSpeed && (
                            <div className="flex justify-between text-gray-400">
                              <span>傳播速度</span>
                              <span className="text-white font-medium">{selectedShockwave.propagationSpeed.toFixed(1)} km/h</span>
                            </div>
                          )}
                          {selectedShockwave.waveDirection && (
                            <div className="flex justify-between text-gray-400">
                              <span>波向</span>
                              <span className="text-white font-medium">
                                {selectedShockwave.waveDirection === 'upstream' ? '上游' : '下游'}
                              </span>
                            </div>
                          )}
                          {selectedShockwave.speedDrop !== undefined && (
                            <div className="flex justify-between text-gray-400">
                              <span>速度下降</span>
                              <span className="text-red-400 font-medium">{selectedShockwave.speedDrop.toFixed(1)} km/h</span>
                            </div>
                          )}
                          {selectedShockwave.queueGrowthRate !== undefined && (
                            <div className="flex justify-between text-gray-400">
                              <span>隊列增長率</span>
                              <span className="text-orange-400 font-medium">{selectedShockwave.queueGrowthRate.toFixed(0)} 車/小時</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* 座標資訊 */}
                      <div className="bg-slate-700/50 rounded-xl p-3 border border-white/5">
                        <div className="text-xs text-gray-500 mb-1">座標位置</div>
                        <div className="text-xs font-mono text-cyan-400">
                          {selectedShockwave.lat.toFixed(6)}, {selectedShockwave.lng.toFixed(6)}
                        </div>
                      </div>
                    </div>
                  );
                })()
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-4 border border-white/10">
                    <ExclamationTriangleIcon className="w-8 h-8 text-gray-500" />
                  </div>
                  <p className="text-gray-400 font-medium">請選擇衝擊波</p>
                  <p className="text-gray-500 text-xs mt-1">點擊列表或地圖上的標記</p>
                </div>
              )}
            </div>
          </div>

          {/* 主要內容區域 */}
          <div className="xl:col-span-3">
            {selectedView === 'predictions' ? (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
                <MTSTNetPredictor />
              </div>
            ) : (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden" style={{ height: '750px' }}>
                <div className="p-4 border-b border-white/10">
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-white flex items-center">
                      <MapIcon className="w-5 h-5 mr-2 text-blue-400" />
                      即時交通監控地圖
                    </h2>

                    <div className="flex items-center space-x-4 text-sm">
                      <div className="flex items-center space-x-2 text-gray-400">
                        <div className="w-3 h-3 bg-green-500 rounded-full shadow-lg shadow-green-500/50" />
                        <span>順暢</span>
                      </div>
                      <div className="flex items-center space-x-2 text-gray-400">
                        <div className="w-3 h-3 bg-yellow-500 rounded-full shadow-lg shadow-yellow-500/50" />
                        <span>壅塞</span>
                      </div>
                      <div className="flex items-center space-x-2 text-gray-400">
                        <div className="w-3 h-3 bg-red-500 rounded-full shadow-lg shadow-red-500/50" />
                        <span>阻塞</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div style={{ height: 'calc(100% - 73px)' }}>
                  <TrafficMap
                    center={{ lat: 25.0330, lng: 121.5654 }}
                    trafficData={trafficData}
                    shockwaves={shockwaves as any}
                    predictions={predictions}
                    showTrafficLayer={true}
                    showShockwaveOverlay={true}
                    zoom={9}
                    selectedShockwaveId={selectedShockwaveId}
                    onShockwaveClick={(id) => setSelectedShockwaveId(id)}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* RAG 聊天機器人彈窗 */}
      {chatbotOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setChatbotOpen(false)}
          />
          <div className="relative w-full max-w-2xl max-h-[80vh] bg-slate-800 rounded-2xl border border-white/10 shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-white/10 bg-gradient-to-r from-blue-600/20 to-purple-600/20">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                  <ChatBubbleLeftRightIcon className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-white font-semibold">AI 智能助手</h3>
                  <p className="text-xs text-gray-400">管理者模式</p>
                </div>
              </div>
              <button
                onClick={() => setChatbotOpen(false)}
                className="p-2 rounded-full hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <div className="h-[60vh]">
              <RAGChatbot
                isOpen={true}
                onClose={() => setChatbotOpen(false)}
                position="embedded"
                trafficData={trafficData}
                shockwaves={shockwaves}
                predictions={predictions}
              />
            </div>
          </div>
        </div>
      )}

      {/* 自定義滾動條樣式 */}
      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }
      `}</style>
    </div>
  );
};

export default ControlCenter;
