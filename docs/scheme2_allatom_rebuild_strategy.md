# scheme2 全原子重建策略分析

> 日期：2026-07-20
> 触发问题：amber14 OL3 全原子精修 `e0=3.9e16, e1=48439`（正得离谱）,
> 最小化没把结构拉到能量井底。根因诊断 + 方案对比如下。

## 1. 现状

### 1.1 数据现象

| 项 | 值 | 状态 |
|---|---|---|
| `e0` (初始能量) | 3.9e16 kJ/mol | 天文正数,LJ 排斥爆炸 |
| `e1` (最小化后) | 48439 kJ/mol | 仍正,没进能量井 |
| 原子重叠 (<0.5Å) | 20+ 对 | 物理结构有病 |
| `n_h` | 330 | ✓ H map 修复生效 |
| `max_p_drift` | 3.2Å < 阈值 3.5Å | ✓ P restraint 钉住 |

### 1.2 根因诊断（已验证）

`allatom_reconstruct.py` 的手写模板在 5.9Å CG P-P 间距下**必然产生原子重叠**：

- 模板里每个残基的骨架原子沿 backbone（b 方向）展开到 x=5.5Å（C1'）
- 相邻 P-P 间距 5.9Å（CG 的 bond_length）
- 残基 i 的 C1'（x=5.5）几乎贴着下游 P[i+1]（x=5.9）,下游 OP1 往上游伸 0.5Å → 重叠
- 旋转碱基模板（±180° 全扫过）也清不干净——剩下的冲突是糖环边界原子（C1'/O3'/C3'）撞相邻残基 OP1/碱基,跟碱基朝向无关
- **物理上无解**：C1'-P 真实键长 ~5Å,5.9Å 间距下挤死

这是 `allatom_reconstruct` 的放置策略跟 CG 5.9 间距不兼容,**不是 tweak 能解决的**。

## 2. 5.9 的语义（必须先澄清）

5.9Å 在代码库里是 **CG 建模常数**：

- `validate_stereochemistry.py` 注释："键长参数（粗粒化：P-P 距离）5.9"
- `physics_bridge.py` 注释："3.4 Å P-P, 5.9 Å P-P next-neighbor"
- 全库 100+ 处用 `bond_length=5.9`（constraint_solver/circrna_diffusion/torusfold/torusfold_moe/所有 train_*.py/scheme8/scheme10/…）
- **13 万条训练数据都在 5.9 下标定**

5.9 **不是**物理 A-form RNA 的 P-P 距离（真实 ~6.5-7.5Å）。
5.9 **也不是** O3'-P 单键键长（那个是 ~1.6Å）。

它在 CG 层自洽——所有 scheme 共享这个几何基准。
改 6.9 = 推翻 13 万数据 + 全库 scheme 重标定,**paper 级建模决定,不该轻易动**。

## 3. 方案对比

### 方案 A：P 对齐 CG（amber 是 CG 骨架约束下的全原子松弛）

#### 技术路线

1. 保留 `allatom_reconstruct` 的 CG P 锚点
2. 改 amber 的 P restraint 更强（当前 K=1000 → 5000+）把 P 完全钉死
3. 接受残基内部几何被硬挤 → 靠 amber 最小化硬拉
4. 或用更激进的方案：amber 前先做"clash 消解"预处理（软推开重叠原子到 >2.5Å）,再最小化

#### paper framing

> "CG-guided all-atom refinement: 在 CG 求解的 P 云约束下,对 circRNA 全原子结构进行立体化学松弛"

卖点：CG 是输入契约,amber 只做精修;端到端可追溯。

#### 风险

- **e1 可能仍是正值**：残基内部几何在 5.9 下天然挤,amber 硬拉不到物理合理值
- **clash 消解引入新变量**：soft-push 后结构合理性存疑,审稿人可能挑
- **物理诚实性**：5.9Å P-P 本身就是建模近似,强行在真实力场下松弛等于逼 amber 接受一个"假"的骨架
- 写进 paper 的方法学容易被质疑

### 方案 B：P 不对齐（amber 独立全原子建模 + CG 二级结构引导）

#### 技术路线

1. **放弃 `allatom_reconstruct` 的 CG P 锚点**
2. 用真实 A-form RNA 几何（P-P ~7Å,键长键角二面角全套 amber 参数）从序列+二级结构构建全原子坐标
3. 实现方式候选：
   - **B1 装 modeRNA**（Python 包,从序列+二级结构建模）
   - **B2 装 biotite 或 biopython**（用内置 A-form 模板）
   - **B3 用 openmm + RNA.OL3.xml 残基模板自己写坐标生成器**（最灵活但工作量大）
   - **B4 云端走 Rosetta FARFAR2 / ModeRNA server**（本地没,但云端能跑）
4. 生成结构直接喂 amber 精修（无需 clash 消解,P-P ~7Å 天然不挤）
5. **CG P 云改用途**：
   - 原来 `coords_cg` 用于 `bsj_after`（闭环质量）+ confidence → 改用全原子 P 算
   - ViennaRNA pairs 仍来自 CG（二级结构输入,与 P 坐标无关）
6. 端到端管线语义变化：从 "CG→全原子" 变成 "CG(提供二级结构+拓扑)+全原子独立建模→精修"

#### paper framing

> "Physics-based all-atom construction from sequence + secondary structure: 
> ViennaRNA pairs 引导,真实 A-form RNA 几何建模,amber14 OL3 精修"

卖点：物理合理,能量转负,paper 方法学无可挑剔。

#### 风险

- **CG P 云失效**：原 scheme2 管线里"CG 求解→全原子"这条语义链断了,scheme2 不再完全由 CG 驱动,而是混合（CG 出二级结构+形状先验,全原子独立建模）
- **工具依赖**：B1/B2/B4 都引入新依赖（本地装包或云端服务）,增加运维成本
- **实现量**：B3 自己写坐标生成器,工作量大,得对 A-form RNA 几何很熟
- **端到端验证**：新工具生成的结构,跟原来的 10 免疫指标对齐关系要重验

## 4. 倾向与理由

**我倾向方案 B,理由：**

1. **物理诚实性**：paper 级方案必须能过同行评审。方案 A 的 e1 正/重叠问题很难干净解决,写出去会被挑
2. **方案 A 的根因是 5.9 间距的建模近似**,CG 层没问题但传到全原子层就是病态
3. **方案 B 的物理合理性高**：A-form RNA 真实几何下,残基内部无冲突,amber 能量天然转负
4. **CG 仍有价值**：方案 B 里 CG 提供二级结构+形状先验,不是完全丢掉,而是换了一种更合理的耦合方式

**具体建议实现路径：B3 + 云端验证**

- 本地用 openmm 8.5 内置的 RNA 残基模板 + 自己写坐标生成器（从序列+A-form 参数生成）
- 云端用 ModeRNA / Rosetta 做对照验证
- CG P 云的下游消费（闭环质量+confidence）改用全原子 P 算

## 5. 待拍板点

1. **方案 A vs B 选哪个**（核心选择）
2. **如果选 B,工具链选哪个**：B1 modeRNA / B2 biotite / B3 自写坐标生成器 / B4 云端模式
3. **CG P 云下游如何改**：闭环质量+confidence 改用全原子 P 算,scheme2 的管线语义更新
4. **是否需要云端验证**：本地生成的结构用云端 ModeRNA / Rosetta 做对照

## 6. 已落地的修复（无论方案 A/B 都保留）

- `immune_heuristic.py` 接入 predictor（10 个免疫指标几何启发式）
- `_RNA_H_MAP` bug 修复（H1/H2/H3/H5/H6 按碱基区分）
- `_add_rna_bonds` 加 H-重原子键
- `implicitSolvent=True` kwarg 冲突（去掉,force field 自带 GBSA）
- 全局参数 `k` 跨 force 撞名 → `k_prestraint`/`k_pairdist`/`k_aform`
- `P_RESTRAINT_K` 10→1000,`P_MAX_DRIFT_A` 2.0→3.5（跟 K 配）

这些是真 bug 修复 + 参数标定,跟重建策略无关,**任何方案都保留**。
