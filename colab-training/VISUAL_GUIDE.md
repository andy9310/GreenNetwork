# 🎨 Visual Guide - Colab Training Repository

**A visual walkthrough of the GreenNetwork Colab training environment**

---

## 📊 Repository Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  COLAB-TRAINING REPOSITORY                  │
│                                                             │
│  🎯 Main Entry Point:                                       │
│     GreenNetwork_Colab_Training.ipynb                       │
│                                                             │
│  📚 Documentation:                                          │
│     ├── README.md (Full guide)                              │
│     ├── QUICK_START.md (5-min guide)                        │
│     ├── SETUP_SUMMARY.md (Overview)                         │
│     ├── INDEX.md (Navigation)                               │
│     └── VISUAL_GUIDE.md (This file)                         │
│                                                             │
│  ⚙️ Configurations:                                         │
│     ├── tiny_network.json (20 nodes)                        │
│     ├── small_network.json (50 nodes) ⭐                    │
│     ├── medium_network.json (100 nodes)                     │
│     └── large_network.json (200 nodes)                      │
│                                                             │
│  🔬 Source Code:                                            │
│     ├── env.py (Environment)                                │
│     ├── agent.py (DQN Agent)                                │
│     ├── cluster.py (Clustering)                             │
│     ├── algorithm.py (Link Deactivation)                    │
│     └── utils.py (Visualization)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Training Workflow

```
┌──────────────┐
│   START      │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│  1. Open Notebook        │
│     in Google Colab      │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  2. Install Dependencies │
│     (30 seconds)         │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  3. Setup Environment    │
│     - Detect GPU         │
│     - Mount Drive        │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  4. Choose Config        │
│     tiny/small/medium/   │
│     large                │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  5. Initialize           │
│     - Create Env         │
│     - Create Agent       │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  6. TRAINING LOOP        │
│     ┌─────────────────┐  │
│     │ For each episode│  │
│     │   ├─ Reset env  │  │
│     │   ├─ Act        │  │
│     │   ├─ Learn      │  │
│     │   └─ Log        │  │
│     └─────────────────┘  │
│     [Live Plots Update]  │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  7. Results & Analysis   │
│     - Summary Stats      │
│     - Interactive Plots  │
│     - Clustering Info    │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  8. Model Evaluation     │
│     - Test best model    │
│     - Greedy policy      │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  9. Export Results       │
│     - Download CSV       │
│     - Download Model     │
└──────┬───────────────────┘
       │
       ▼
┌──────────────┐
│     END      │
└──────────────┘
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌────────────┐ │
│  │   Agent     │◄────►│ Environment │◄────►│  Cluster   │ │
│  │   (DQN)     │      │    (SDN)    │      │  Manager   │ │
│  └─────────────┘      └─────────────┘      └────────────┘ │
│         │                     │                    │        │
│         │                     │                    │        │
│         ▼                     ▼                    ▼        │
│  ┌─────────────┐      ┌─────────────┐      ┌────────────┐ │
│  │   Replay    │      │   Traffic   │      │  DP-means  │ │
│  │   Buffer    │      │  Generator  │      │  K-means   │ │
│  └─────────────┘      └─────────────┘      └────────────┘ │
│         │                     │                    │        │
│         └─────────────────────┴────────────────────┘        │
│                              │                              │
│                              ▼                              │
│                    ┌──────────────────┐                     │
│                    │   Link Manager   │                     │
│                    │  (Deactivation)  │                     │
│                    └──────────────────┘                     │
│                              │                              │
│                              ▼                              │
│                    ┌──────────────────┐                     │
│                    │   Visualizer     │                     │
│                    │  (Colab Utils)   │                     │
│                    └──────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Training Progress Visualization

```
Episode 1-100: Exploration Phase
████░░░░░░░░░░░░░░░░ 20%
Reward: -5 to +2
Energy: 20-40%
Learning: Random actions

Episode 100-300: Learning Phase
████████████░░░░░░░░ 60%
Reward: +2 to +10
Energy: 40-65%
Learning: Improving policy

Episode 300-500: Convergence Phase
████████████████████ 100%
Reward: +10 to +15
Energy: 70-80% ✅
Learning: Stable policy
```

---

## 🎯 Network Configurations Comparison

```
┌─────────────┬──────────┬──────────┬──────────┬──────────┐
│   Config    │   Tiny   │  Small   │  Medium  │  Large   │
├─────────────┼──────────┼──────────┼──────────┼──────────┤
│ Nodes       │    20    │    50    │   100    │   200    │
│ Edges       │    40    │   100    │   500    │  2000    │
│ Hosts       │     8    │    20    │    40    │    80    │
│ Episodes    │   100    │   500    │  1000    │  2000    │
│ Time (GPU)  │  2-5 min │ 5-10 min │ 15-30min │ 1-2 hrs  │
│ Use Case    │   Test   │  Develop │  Verify  │ Research │
└─────────────┴──────────┴──────────┴──────────┴──────────┘

Recommendation: Start with "Small" ⭐
```

---

## 📊 Metrics Dashboard Layout

```
┌────────────────────────────────────────────────────────────┐
│                    TRAINING DASHBOARD                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │  Episode Reward  │  │  Energy Saving   │              │
│  │                  │  │                  │              │
│  │      ╱╲          │  │        ╱─────    │              │
│  │     ╱  ╲    ╱╲   │  │    ╱──╱          │              │
│  │  ╱─╱    ╲──╱  ╲  │  │ ╱─╱              │              │
│  │                  │  │                  │              │
│  └──────────────────┘  └──────────────────┘              │
│                                                            │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │     Latency      │  │  SLA Violations  │              │
│  │                  │  │                  │              │
│  │  ────────────    │  │  ╲               │              │
│  │              ╲   │  │   ╲              │              │
│  │               ╲  │  │    ╲──────────   │              │
│  │                  │  │                  │              │
│  └──────────────────┘  └──────────────────┘              │
│                                                            │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │  Active Links    │  │   Utilization    │              │
│  │                  │  │                  │              │
│  │  ╲               │  │  ─────────────   │              │
│  │   ╲              │  │                  │              │
│  │    ╲─────────    │  │                  │              │
│  │                  │  │                  │              │
│  └──────────────────┘  └──────────────────┘              │
└────────────────────────────────────────────────────────────┘
```

---

## 🔄 Agent Learning Process

```
┌─────────────────────────────────────────────────────────┐
│              DQN LEARNING CYCLE                         │
└─────────────────────────────────────────────────────────┘

1. OBSERVE
   ┌──────────────────┐
   │ Network State    │
   │ - Traffic load   │
   │ - Link utils     │
   │ - Cluster info   │
   └────────┬─────────┘
            │
            ▼
2. DECIDE
   ┌──────────────────┐
   │ Q-Network        │
   │ Selects action:  │
   │ - Thresholds     │
   │ - Inter-cluster  │
   └────────┬─────────┘
            │
            ▼
3. ACT
   ┌──────────────────┐
   │ Environment      │
   │ - Deactivate     │
   │ - Route traffic  │
   │ - Measure        │
   └────────┬─────────┘
            │
            ▼
4. REWARD
   ┌──────────────────┐
   │ Calculate        │
   │ Energy - Penalty │
   │ (latency + SLA)  │
   └────────┬─────────┘
            │
            ▼
5. LEARN
   ┌──────────────────┐
   │ Update Q-Network │
   │ - Sample batch   │
   │ - Compute loss   │
   │ - Backprop       │
   └────────┬─────────┘
            │
            └──────────► (Repeat)
```

---

## 🎮 Action Space Structure

```
┌─────────────────────────────────────────────────────────┐
│                  HIERARCHICAL ACTIONS                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Per-Cluster Thresholds (9 bins each):                 │
│                                                         │
│  Cluster 0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
│  Cluster 1: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
│  Cluster 2: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
│  ...                                                    │
│  Cluster K: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
│                                                         │
│  Inter-Cluster Links: [2, 3, 4, 5] (min to keep)       │
│                                                         │
│  Total Actions = 9^K × 4                                │
│                                                         │
│  Example (K=3): 9³ × 4 = 2,916 actions                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🌐 Network Topology Example

```
Small Network (50 nodes, 100 edges)

        Region 0              Region 1              Region 2
    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
    │             │       │             │       │             │
    │  ●─●─●─●    │       │  ●─●─●─●    │       │  ●─●─●─●    │
    │  │ │ │ │    │       │  │ │ │ │    │       │  │ │ │ │    │
    │  ●─●─●─●    │◄─────►│  ●─●─●─●    │◄─────►│  ●─●─●─●    │
    │  │ │ │ │    │       │  │ │ │ │    │       │  │ │ │ │    │
    │  ●─●─●─●    │       │  ●─●─●─●    │       │  ●─●─●─●    │
    │             │       │             │       │             │
    └─────────────┘       └─────────────┘       └─────────────┘
         ▲                     ▲                     ▲
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                    Inter-cluster links

    ● = Switch/Router
    ─ = Active link
    ╌ = Inactive link (energy saved)
    ◆ = Host
```

---

## 📊 Performance Metrics Over Time

```
Energy Saving (%)
100 ┤                                    ╭─────────
 90 ┤                              ╭────╯
 80 ┤                         ╭────╯              ← Target: 70-80%
 70 ┤                    ╭────╯
 60 ┤               ╭────╯
 50 ┤          ╭────╯
 40 ┤     ╭────╯
 30 ┤╭────╯
 20 ┤╯
  0 └┴────┴────┴────┴────┴────┴────┴────┴────┴────►
    0   50  100  150  200  250  300  350  400  500
                      Episodes

SLA Violations (%)
 30 ┤╮
 25 ┤│╲
 20 ┤│ ╲
 15 ┤│  ╲                                          ← Target: <10%
 10 ┤│   ╲────╮
  5 ┤│        ╰────────────────────────────────
  0 └┴────┴────┴────┴────┴────┴────┴────┴────┴────►
    0   50  100  150  200  250  300  350  400  500
                      Episodes
```

---

## 🎯 Success Indicators

```
┌─────────────────────────────────────────────────────────┐
│              TRAINING SUCCESS CHECKLIST                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ Reward increasing over time                         │
│  ✅ Energy saving reaches 70%+                          │
│  ✅ SLA violations below 10%                            │
│  ✅ Latency stays low (10-20ms)                         │
│  ✅ Loss decreasing                                     │
│  ✅ Epsilon decaying to 0.05                            │
│  ✅ Utilization in target range                         │
│  ✅ Clustering stable                                   │
│  ✅ No overload warnings                                │
│  ✅ Models saving successfully                          │
│                                                         │
└─────────────────────────────────────────────────────────┘

If ALL checked: 🎉 Training Successful!
```

---

## 🔧 Troubleshooting Flowchart

```
                    ┌─────────────┐
                    │   Problem?  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Out of Memory │  │  Slow Training│  │ Poor Results  │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ • Enable GPU  │  │ • Enable GPU  │  │ • Train longer│
│ • Use smaller │  │ • Reduce size │  │ • Adjust LR   │
│   network     │  │ • Reduce      │  │ • Tune params │
│ • Reduce      │  │   episodes    │  │ • Check config│
│   batch size  │  │               │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## 📱 Quick Reference Card

```
╔═══════════════════════════════════════════════════════════╗
║              COLAB TRAINING QUICK REFERENCE               ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  🚀 START:  Open notebook → Run all cells                ║
║                                                           ║
║  ⚙️ CONFIG: CONFIG_NAME = 'small_network'                ║
║                                                           ║
║  🎯 TARGET: Energy 70%+, SLA <10%, Latency 10-15ms       ║
║                                                           ║
║  💾 SAVE:   Auto-saves to Google Drive                   ║
║                                                           ║
║  📊 PLOTS:  Update every 10 episodes                     ║
║                                                           ║
║  ⏱️ TIME:   Small=5-10min, Medium=15-30min, Large=1-2hr ║
║                                                           ║
║  🔧 CUSTOM: Edit config dict in Cell 7                   ║
║                                                           ║
║  🆘 HELP:   Check QUICK_START.md troubleshooting         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## 🎓 Learning Progression

```
Level 1: Beginner (Day 1)
├─ Read QUICK_START.md
├─ Run tiny_network
├─ Understand basic metrics
└─ View live plots

Level 2: Intermediate (Week 1)
├─ Train small_network
├─ Experiment with configs
├─ Try different traffic modes
└─ Analyze results

Level 3: Advanced (Month 1)
├─ Train large_network
├─ Modify hyperparameters
├─ Compare algorithms
└─ Optimize performance

Level 4: Expert (Month 2+)
├─ Modify source code
├─ Implement new features
├─ Conduct experiments
└─ Publish results
```

---

## 🎨 Color Coding Legend

```
📘 Blue   = Documentation files
📗 Green  = Configuration files
📙 Orange = Source code files
📕 Red    = Important warnings
📓 Purple = Optional/Advanced
```

---

**Visual Guide Complete! 🎨**

Use this guide alongside the main documentation for a comprehensive understanding of the system.

---

**Last Updated**: 2025-10-11  
**Version**: 1.0
