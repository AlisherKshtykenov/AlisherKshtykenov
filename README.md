<!-- Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=230&color=0:2b1408,55:7a3b16,100:d97742&text=Alisher%20Kshtykenov&fontColor=fff4e8&fontSize=52&fontAlignY=36&desc=ML%20%C2%B7%20AI%20agents%20%C2%B7%20Automation%20%C2%B7%20macOS%20tools&descSize=18&descAlignY=58&animation=fadeIn" alt="Alisher Kshtykenov" width="100%">
</p>

<p align="center">
  <a href="https://github.com/AlisherKshtykenov">
    <img src="https://readme-typing-svg.demolab.com/?font=JetBrains+Mono&weight=500&size=20&duration=3200&pause=900&color=D97742&center=true&vCenter=true&width=640&lines=Training+models+from+scratch+%E2%80%94+and+knowing+why+they+work;Building+AI+agents+that+run+their+own+experiments;Automating+the+boring+parts+with+n8n+%2B+Claude+Code;Shipping+tiny+native+macOS+apps+for+fun+%F0%9F%A6%80" alt="Typing intro">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/aka-КАШТАН_🌰-7a3b16?style=flat-square" alt="aka КАШТАН">
  <img src="https://img.shields.io/badge/based_in-Kazakhstan_🇰🇿-d97742?style=flat-square" alt="Kazakhstan">
  <img src="https://komarev.com/ghpvc/?username=AlisherKshtykenov&style=flat-square&color=d97742&label=profile+views" alt="Profile views">
</p>

---

### 👋 About me

```python
class Alisher:
    role      = "ML student · AI-agent & automation builder"
    focus     = ["machine learning from first principles", "LLM agents", "n8n automation"]
    currently = "building a self-driving research lab: Claude Code + n8n"
    principle = "smallest working whole first, then grow it step by step"
    languages = {"code": ["Python", "Swift", "TypeScript", "Java", "Rust"], "human": ["Russian", "English"]}
```

- 🧠 I like to **write things by hand before calling the library**: logistic regression, MLPs and CNN blocks in NumPy and PyTorch, checked against sklearn and torch.
- 🤖 I build **agent systems**: labs and "factories" where Claude Code proposes hypotheses, runs experiments, commits or rolls back, and reports to Telegram.
- ⚙️ I automate workflows with **n8n**, Python and webhooks. Every workflow is tested and keeps its secrets out of git.
- 🍎 For fun, I make **tiny native macOS apps** in pure Swift, with no Xcode project: widgets, live wallpapers, and a desktop crab.

---

### 🚀 Featured projects

<table>
  <tr>
    <td width="50%" valign="top">
      <h4>🧪 <a href="https://github.com/AlisherKshtykenov/n8n-laboratory">n8n-laboratory</a></h4>
      A research lab where <b>Claude Code</b> works on any task in "hypothesis → run → commit or roll back" cycles, with <b>n8n</b> as the conveyor and a live board. First run on MNIST: val 0.91 → <b>0.98</b>, audited, reported to Telegram.
      <br><br>
      <img src="https://img.shields.io/badge/n8n-EA4B71?style=flat-square&logo=n8n&logoColor=white">
      <img src="https://img.shields.io/badge/Claude_Code-D97757?style=flat-square&logo=anthropic&logoColor=white">
      <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">
      <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white">
    </td>
    <td width="50%" valign="top">
      <h4>‹› <a href="https://github.com/AlisherKshtykenov/code-recall">CODE RECALL</a></h4>
      LeetCode for <i>your own</i> code. You write algorithms and ML from scratch again from memory in a browser mini-IDE, graded by pytest and scheduled with <b>FSRS</b> spaced repetition. Includes a visual task builder and semantic diff.
      <br><br>
      <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
      <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">
      <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white">
      <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white">
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h4>🧠 <a href="https://github.com/AlisherKshtykenov/cifar-cnn-ablation">cifar-cnn-ablation</a></h4>
      12 CNNs trained from scratch, one technique changed at a time, from a 53% MLP to a <b>92.6% ResNet-9</b> on CIFAR-10 (69.6% top-1 on CIFAR-100). Every gain is measured: BatchNorm + Dropout +7.6 pp, residuals +2.5 pp.
      <br><br>
      <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
      <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
      <img src="https://img.shields.io/badge/Apple_MPS-000000?style=flat-square&logo=apple&logoColor=white">
    </td>
    <td width="50%" valign="top">
      <h4>🎯 <a href="https://github.com/AlisherKshtykenov/instance-segmentation-benchmark">instance-segmentation-benchmark</a></h4>
      Mask R-CNN v2 vs YOLO26-seg on COCO val2017: mask AP with pycocotools, FPS on GPU and CPU, and a speed vs accuracy map across input sizes. <b>AP 50.3 at 1.3 FPS vs AP 40.6 at 28 FPS</b>, and which one to ship where.
      <br><br>
      <img src="https://img.shields.io/badge/torchvision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
      <img src="https://img.shields.io/badge/YOLO-111F68?style=flat-square">
      <img src="https://img.shields.io/badge/COCO-d97742?style=flat-square">
    </td>
  </tr>
</table>

### 🔬 ML research

| Project | What it answers | Headline |
|---|---|---|
| 🧠 [cifar-cnn-ablation](https://github.com/AlisherKshtykenov/cifar-cnn-ablation) | What is each CNN technique actually worth? | **92.6%** CIFAR-10 · **69.6%** CIFAR-100 |
| ✨ [cifar10-image-denoising](https://github.com/AlisherKshtykenov/cifar10-image-denoising) | Autoencoders, U-Net or DnCNN: what actually removes noise, and which idea matters? | **28.45 dB** PSNR, skips worth **+2.8 dB** |
| 🎯 [instance-segmentation-benchmark](https://github.com/AlisherKshtykenov/instance-segmentation-benchmark) | Two-stage or one-stage segmentation, and when? | AP **50.3** vs **28 FPS** |
| 🌡️ [pinn-heat-equation](https://github.com/AlisherKshtykenov/pinn-heat-equation) | Forward PINN vs finite differences, and an inverse PINN that recovers an unknown coefficient | α recovered to **0.004%** |
| 🌊 [pinn-viscoacoustic-wave](https://github.com/AlisherKshtykenov/pinn-viscoacoustic-wave) | 2D wave equation in a depth-dependent medium: what does dropping the coefficient-gradient terms cost? | **2.4%** vs **7.9%** error |
| 🧩 [xpinn-material-interface](https://github.com/AlisherKshtykenov/xpinn-material-interface) | One network per rock layer, glued by displacement and stress continuity: how hard to enforce it, and do Fourier features help? | **0.29%** error, Fourier scale matters |
| 🎮 [gymnasium-rl-benchmark](https://github.com/AlisherKshtykenov/gymnasium-rl-benchmark) | Which RL algorithm solves control tasks reliably, and at what sample cost? | LunarLander **261**, own Deep CEM and REINFORCE |
| 📋 [tabular-ml-studies](https://github.com/AlisherKshtykenov/tabular-ml-studies) | RF vs SVM with error costs, an MLP in pure NumPy, PLE nets vs gradient boosting | bootstrap-tested choices |
| 🏀 [nba-elo-eda](https://github.com/AlisherKshtykenov/nba-elo-eda) | 60 years of NBA games: stratification, confounding, χ², PCA | 4-part EDA |

<details>
<summary><b>🍎 Native macOS toys</b> (pure Swift, one file each, no Xcode project)</summary>
<br>

| | Project | What it does |
|---|---|---|
| ▦ | [block-widgets](https://github.com/AlisherKshtykenov/block-widgets) | glass desktop widgets: clock, weather, Pomodoro, network, git status, **live ML-training progress** |
| ❄️ | [snow-desktop](https://github.com/AlisherKshtykenov/snow-desktop) | animated wallpaper: a snowy cabin, campfire and falling snow, paused when hidden |
| 🎞 | [live-wallpaper](https://github.com/AlisherKshtykenov/live-wallpaper) | any MP4 as a looping, battery-friendly desktop wallpaper |
| 🦀 | [clawd-pet](https://github.com/AlisherKshtykenov/clawd-pet) | the Claude Code crab walking around your desktop |

</details>

<details>
<summary><b>🌐 Full-stack</b></summary>
<br>

| Project | |
|---|---|
| [spring-react-fullstack](https://github.com/AlisherKshtykenov/spring-react-fullstack) | Spring Boot 3 REST API with JWT, Spring Data REST, OpenAPI and tests, plus six React 19 / TypeScript apps |

</details>

---

### 🛠 Tech stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=py,pytorch,sklearn,fastapi,swift,rust,java,spring,react,ts,js,docker,sqlite,supabase,vercel,cloudflare,git,bash&perline=9&theme=dark" alt="Tech stack">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/n8n-EA4B71?style=for-the-badge&logo=n8n&logoColor=white">
  <img src="https://img.shields.io/badge/Claude_Code-D97757?style=for-the-badge&logo=anthropic&logoColor=white">
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white">
  <img src="https://img.shields.io/badge/Telegram_Bots-26A5E4?style=for-the-badge&logo=telegram&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
</p>

---

### 📈 Activity

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AlisherKshtykenov/AlisherKshtykenov/output/streak-dark.svg">
    <img alt="Contribution streak" src="https://raw.githubusercontent.com/AlisherKshtykenov/AlisherKshtykenov/output/streak.svg" width="495">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AlisherKshtykenov/AlisherKshtykenov/output/heatmap-dark.svg">
    <img alt="Contribution heatmap" src="https://raw.githubusercontent.com/AlisherKshtykenov/AlisherKshtykenov/output/heatmap.svg" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AlisherKshtykenov/AlisherKshtykenov/output/graph-dark.svg">
    <img alt="Contributions in the last 30 days" src="https://raw.githubusercontent.com/AlisherKshtykenov/AlisherKshtykenov/output/graph.svg" width="100%">
  </picture>
</p>


<details>
<summary>🇷🇺 <b>Коротко по-русски</b></summary>
<br>

Привет! Я Алишер, ник **КАШТАН**. Изучаю машинное обучение «с нуля», то есть сначала пишу руками и только потом беру библиотеку.
Строю агентные системы на Claude Code и n8n, автоматизирую рутину и для души делаю маленькие нативные приложения для macOS.
Главный проект сейчас — **n8n-laboratory**: лаборатория, где агент сам ставит гипотезы, проверяет их, коммитит или откатывает и присылает отчёт в Telegram.

</details>

<!-- Footer -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=110&section=footer&color=0:2b1408,55:7a3b16,100:d97742" width="100%">
</p>
