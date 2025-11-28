'use client';

import { useRouter } from 'next/router'
import { useState, useEffect, useRef } from 'react'
import {
  ChartBarIcon,
  UserIcon,
  CogIcon,
  ArrowRightIcon,
  BoltIcon,
  ClockIcon,
  SignalIcon,
  CpuChipIcon,
  GlobeAltIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline'

// 動態波紋效果組件
const ShockwaveRipple = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {[...Array(4)].map((_, i) => (
        <div
          key={i}
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-orange-400/30"
          style={{
            width: `${200 + i * 150}px`,
            height: `${200 + i * 150}px`,
            animation: `ripple 4s ease-out infinite`,
            animationDelay: `${i * 1}s`,
          }}
        />
      ))}
    </div>
  )
}

// 流動粒子背景
const ParticleBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    // 粒子系統 - 模擬車流
    const particles: Array<{
      x: number
      y: number
      speed: number
      size: number
      opacity: number
      lane: number
    }> = []

    // 初始化粒子
    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        speed: 1 + Math.random() * 2,
        size: 2 + Math.random() * 3,
        opacity: 0.3 + Math.random() * 0.4,
        lane: Math.floor(Math.random() * 5)
      })
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      particles.forEach((p) => {
        // 繪製粒子（模擬車輛）
        ctx.beginPath()
        ctx.fillStyle = `rgba(249, 115, 22, ${p.opacity})`
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fill()

        // 繪製尾跡
        ctx.beginPath()
        ctx.strokeStyle = `rgba(249, 115, 22, ${p.opacity * 0.3})`
        ctx.lineWidth = p.size * 0.5
        ctx.moveTo(p.x, p.y)
        ctx.lineTo(p.x - p.speed * 10, p.y)
        ctx.stroke()

        // 移動粒子
        p.x += p.speed
        if (p.x > canvas.width + 10) {
          p.x = -10
          p.y = Math.random() * canvas.height
        }
      })

      requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none opacity-40"
    />
  )
}

// 即時數據儀表板組件
const LiveDataPanel = ({ stats }: { stats: any }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {[
        { label: '監測站點', value: stats.monitoringStations, icon: GlobeAltIcon, color: 'from-blue-500 to-cyan-500' },
        { label: '即時預測', value: '10-30分', icon: ClockIcon, color: 'from-green-500 to-emerald-500' },
        { label: '系統準確度', value: '87%', icon: ShieldCheckIcon, color: 'from-purple-500 to-pink-500' },
        { label: 'AI 引擎', value: 'MT-STNet', icon: CpuChipIcon, color: 'from-orange-500 to-red-500' },
      ].map((item, index) => (
        <div
          key={index}
          className="relative group"
          style={{ animationDelay: `${index * 0.1}s` }}
        >
          <div className={`absolute inset-0 bg-gradient-to-r ${item.color} rounded-xl blur-xl opacity-20 group-hover:opacity-40 transition-opacity`} />
          <div className="relative bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-4 hover:bg-white/20 transition-all hover:scale-105 hover:-translate-y-1">
            <item.icon className="w-6 h-6 text-white/80 mb-2" />
            <div className="text-2xl font-bold text-white">{item.value}</div>
            <div className="text-sm text-white/60">{item.label}</div>
          </div>
        </div>
      ))}
    </div>
  )
}

// 道路動畫組件
const HighwayAnimation = () => {
  return (
    <div className="absolute bottom-0 left-0 right-0 h-32 overflow-hidden pointer-events-none">
      <svg className="w-full h-full" viewBox="0 0 1200 100" preserveAspectRatio="none">
        {/* 道路 */}
        <path
          d="M0 80 Q300 60, 600 70 T1200 60"
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="40"
        />
        {/* 中線 */}
        <path
          d="M0 80 Q300 60, 600 70 T1200 60"
          fill="none"
          stroke="rgba(255,255,255,0.3)"
          strokeWidth="2"
          strokeDasharray="20 10"
          className="animate-dash"
        />
        {/* 移動的車輛光點 */}
        {[0, 1, 2, 3, 4].map((i) => (
          <circle
            key={i}
            r="4"
            fill="#f97316"
            className="animate-car"
            style={{
              animationDelay: `${i * 0.8}s`,
            }}
          >
            <animateMotion
              dur="4s"
              repeatCount="indefinite"
              path="M0 80 Q300 60, 600 70 T1200 60"
            />
          </circle>
        ))}
      </svg>
    </div>
  )
}

// 技術標籤組件
const TechBadge = ({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) => (
  <span
    className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-white/10 text-white/80 border border-white/20 backdrop-blur-sm animate-fadeIn"
    style={{ animationDelay: `${delay}s` }}
  >
    {children}
  </span>
)

export default function Home() {
  const router = useRouter()
  const [currentTime, setCurrentTime] = useState<Date | null>(null)
  const [isLoaded, setIsLoaded] = useState(false)
  const [activeFeature, setActiveFeature] = useState(0)
  const [systemStats] = useState({
    activeUsers: 1247,
    monitoringStations: 62,
    activeAlerts: 3,
    systemUptime: '99.8%'
  })

  useEffect(() => {
    setCurrentTime(new Date())
    setIsLoaded(true)

    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    // 自動輪播特色功能
    const featureTimer = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % 4)
    }, 3000)

    return () => {
      clearInterval(timer)
      clearInterval(featureTimer)
    }
  }, [])

  const navigateToDriver = () => router.push('/driver')
  const navigateToAdmin = () => router.push('/admin')

  const features = [
    {
      icon: BoltIcon,
      title: '交通衝擊波偵測',
      description: '基於 LWR 理論與 Rankine-Hugoniot 條件，即時偵測交通壅塞傳播',
      detail: '將地震震波理論創新應用於交通流分析',
      color: 'from-orange-500 to-red-500',
      bgColor: 'bg-orange-500/20'
    },
    {
      icon: CpuChipIcon,
      title: 'MT-STNet 深度學習',
      description: '多任務時空神經網路，整合圖卷積與注意力機制',
      detail: '提前 10-30 分鐘預測交通狀態變化',
      color: 'from-purple-500 to-pink-500',
      bgColor: 'bg-purple-500/20'
    },
    {
      icon: ChartBarIcon,
      title: '雙引擎預測架構',
      description: '深度學習引擎處理正常車流，物理引擎應對突發事件',
      detail: '自動切換最適合的預測模式',
      color: 'from-blue-500 to-cyan-500',
      bgColor: 'bg-blue-500/20'
    },
    {
      icon: SignalIcon,
      title: 'RAG 智慧決策',
      description: '檢索增強生成系統，整合道路幾何知識庫',
      detail: '提供個人化的交通決策建議',
      color: 'from-green-500 to-emerald-500',
      bgColor: 'bg-green-500/20'
    }
  ]

  return (
    <div className={`min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden transition-opacity duration-1000 ${isLoaded ? 'opacity-100' : 'opacity-0'}`}>
      {/* 動態背景 */}
      <ParticleBackground />

      {/* 頂部漸層裝飾 */}
      <div className="absolute top-0 left-0 right-0 h-96 bg-gradient-to-b from-orange-500/10 via-purple-500/5 to-transparent pointer-events-none" />

      {/* 導航列 */}
      <nav className="relative z-50 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center animate-pulse-slow">
                  <BoltIcon className="w-6 h-6 text-white" />
                </div>
                <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg blur-lg opacity-50" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white">時間管理大師</h1>
                <p className="text-xs text-white/50">Highway Traffic Intelligence</p>
              </div>
            </div>

            <div className="flex items-center space-x-6">
              <div className="hidden md:flex items-center space-x-2 text-sm text-white/60">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span>系統運行中</span>
              </div>
              <div className="text-sm text-white/80 font-mono">
                {currentTime ? currentTime.toLocaleString('zh-TW', {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit'
                }) : '--:--:--'}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* 主要內容 */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">

        {/* Hero 區塊 */}
        <div className="text-center mb-16 relative">
          <ShockwaveRipple />

          {/* 技術標籤 */}
          <div className="flex flex-wrap justify-center gap-2 mb-8">
            <TechBadge delay={0.2}>LWR 交通流理論</TechBadge>
            <TechBadge delay={0.3}>Rankine-Hugoniot 條件</TechBadge>
            <TechBadge delay={0.4}>MT-STNet</TechBadge>
            <TechBadge delay={0.5}>RAG 決策系統</TechBadge>
          </div>

          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6 animate-fadeInUp">
            <span className="block mb-2">基於深度學習與物理學融合的</span>
            <span className="bg-gradient-to-r from-orange-400 via-red-400 to-purple-400 bg-clip-text text-transparent animate-gradient">
              未來交通預測 AI 決策系統
            </span>
          </h1>

          <p className="text-lg md:text-xl text-white/70 mb-8 max-w-3xl mx-auto animate-fadeInUp" style={{ animationDelay: '0.2s' }}>
            首創將<span className="text-orange-400 font-semibold">地震震波傳播理論</span>應用於交通壅塞分析，
            結合 <span className="text-purple-400 font-semibold">MT-STNet 時空神經網路</span>，
            實現提前 10-30 分鐘的精準預測預警
          </p>

          {/* 即時數據面板 */}
          <div className="mb-12 animate-fadeInUp" style={{ animationDelay: '0.4s' }}>
            <LiveDataPanel stats={systemStats} />
          </div>
        </div>

        {/* 核心特色輪播 */}
        <div className="mb-16">
          <div className="relative bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 p-8 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 via-transparent to-purple-500/5" />

            <div className="relative grid md:grid-cols-2 gap-8 items-center">
              {/* 左側 - 特色列表 */}
              <div className="space-y-4">
                {features.map((feature, index) => (
                  <div
                    key={index}
                    onClick={() => setActiveFeature(index)}
                    className={`p-4 rounded-xl cursor-pointer transition-all duration-300 ${
                      activeFeature === index
                        ? `bg-gradient-to-r ${feature.color} shadow-lg scale-105`
                        : 'bg-white/5 hover:bg-white/10'
                    }`}
                  >
                    <div className="flex items-center space-x-4">
                      <feature.icon className={`w-8 h-8 ${activeFeature === index ? 'text-white' : 'text-white/60'}`} />
                      <div>
                        <h3 className={`font-semibold ${activeFeature === index ? 'text-white' : 'text-white/80'}`}>
                          {feature.title}
                        </h3>
                        <p className={`text-sm ${activeFeature === index ? 'text-white/90' : 'text-white/50'}`}>
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* 右側 - 詳細說明 */}
              <div className="relative">
                <div className={`absolute inset-0 bg-gradient-to-r ${features[activeFeature].color} rounded-2xl blur-2xl opacity-20`} />
                <div className="relative bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20">
                  <div className={`w-16 h-16 rounded-xl bg-gradient-to-r ${features[activeFeature].color} flex items-center justify-center mb-6`}>
                    {(() => {
                      const Icon = features[activeFeature].icon
                      return <Icon className="w-8 h-8 text-white" />
                    })()}
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-4">
                    {features[activeFeature].title}
                  </h3>
                  <p className="text-white/70 mb-4">
                    {features[activeFeature].description}
                  </p>
                  <p className="text-lg text-white/90 font-medium">
                    {features[activeFeature].detail}
                  </p>

                  {/* 進度指示器 */}
                  <div className="flex space-x-2 mt-6">
                    {features.map((_, i) => (
                      <div
                        key={i}
                        className={`h-1 rounded-full transition-all duration-300 ${
                          i === activeFeature ? 'w-8 bg-white' : 'w-2 bg-white/30'
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 入口選擇 */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {/* 駕駛者入口 */}
          <div
            onClick={navigateToDriver}
            className="group relative cursor-pointer"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl blur-xl opacity-20 group-hover:opacity-40 transition-opacity" />
            <div className="relative bg-white/5 backdrop-blur-xl rounded-2xl p-8 border border-white/10 hover:border-blue-500/50 transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1">
              <div className="flex items-center justify-between mb-6">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform shadow-lg shadow-blue-500/25">
                  <UserIcon className="w-8 h-8 text-white" />
                </div>
                <ArrowRightIcon className="w-6 h-6 text-white/40 group-hover:text-blue-400 group-hover:translate-x-2 transition-all" />
              </div>

              <h2 className="text-2xl font-bold text-white mb-3">駕駛者介面</h2>
              <p className="text-white/60 mb-6">
                即時路況查詢、衝擊波預警通知、智慧路線規劃，讓您的每次出行更安全順暢
              </p>

              <div className="flex flex-wrap gap-2">
                {['即時監控', '震波預警', 'AI 助手'].map((tag) => (
                  <span key={tag} className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-sm">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* 管理者入口 */}
          <div
            onClick={navigateToAdmin}
            className="group relative cursor-pointer"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur-xl opacity-20 group-hover:opacity-40 transition-opacity" />
            <div className="relative bg-white/5 backdrop-blur-xl rounded-2xl p-8 border border-white/10 hover:border-purple-500/50 transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1">
              <div className="flex items-center justify-between mb-6">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform shadow-lg shadow-purple-500/25">
                  <CogIcon className="w-8 h-8 text-white" />
                </div>
                <ArrowRightIcon className="w-6 h-6 text-white/40 group-hover:text-purple-400 group-hover:translate-x-2 transition-all" />
              </div>

              <h2 className="text-2xl font-bold text-white mb-3">管理控制中心</h2>
              <p className="text-white/60 mb-6">
                全面監控交通狀況、MT-STNet 預測分析、AI 決策建議，掌握系統全局
              </p>

              <div className="flex flex-wrap gap-2">
                {['系統監控', 'AI 決策', '流量分析'].map((tag) => (
                  <span key={tag} className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 底部資訊 */}
        <div className="text-center">
          <div className="inline-flex items-center space-x-4 px-6 py-3 bg-white/5 backdrop-blur-md rounded-full border border-white/10">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-white/60 text-sm">62 監測站點運行中</span>
            </div>
            <div className="w-px h-4 bg-white/20" />
            <div className="text-white/60 text-sm">
              涵蓋國道一號、國道三號
            </div>
          </div>
        </div>
      </main>

      {/* 道路動畫 */}
      <HighwayAnimation />

      {/* 頁腳 */}
      <footer className="relative z-10 border-t border-white/10 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between text-white/40 text-sm">
            <p>2024 高速公路智慧交通預警決策支援系統</p>
            <p className="mt-2 md:mt-0">中華民國運輸學會年會</p>
          </div>
        </div>
      </footer>

      {/* 自定義動畫樣式 */}
      <style jsx global>{`
        @keyframes ripple {
          0% {
            transform: translate(-50%, -50%) scale(0.8);
            opacity: 0.6;
          }
          100% {
            transform: translate(-50%, -50%) scale(2);
            opacity: 0;
          }
        }

        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes gradient {
          0%, 100% {
            background-size: 200% 200%;
            background-position: left center;
          }
          50% {
            background-size: 200% 200%;
            background-position: right center;
          }
        }

        @keyframes dash {
          to {
            stroke-dashoffset: -30;
          }
        }

        .animate-fadeInUp {
          animation: fadeInUp 0.8s ease-out forwards;
        }

        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out forwards;
        }

        .animate-gradient {
          animation: gradient 3s ease infinite;
        }

        .animate-dash {
          animation: dash 1s linear infinite;
        }

        .animate-pulse-slow {
          animation: pulse 3s ease-in-out infinite;
        }

        .animate-car {
          filter: drop-shadow(0 0 6px #f97316);
        }
      `}</style>
    </div>
  )
}
