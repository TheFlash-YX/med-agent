# Python复现医疗多Agent临床辅助决策系统流程

本文档用于从零开始用 Python 复现 `medical-multi-agent-system` 项目的核心能力。推荐先复现 Python 版，因为它是本项目中最完整的一条主线：`FastAPI + LangGraph + LangChain/OpenAI + Pydantic + 服务层规则库 + Docker基础设施`。

> 注意：这是学习和面试展示型系统，不应直接用于真实临床诊疗决策。

---

## 1. 先理解系统目标

这个系统要完成一条固定的临床辅助决策 Pipeline：

```text
患者自然语言描述
  -> Intake Agent：接诊，抽取结构化患者信息
  -> Diagnosis Agent：诊断，生成主诊断和鉴别诊断
  -> Treatment Agent：治疗，生成治疗方案并检查药物风险
  -> Coding Agent：编码，生成 ICD-10 和 DRG 信息
  -> Audit Agent：审计，检查 PHI/HIPAA 合规
  -> API 返回完整报告
```

核心设计思想：

- 用多个专业 Agent 拆分复杂任务，而不是一个大 Prompt 做所有事。
- 用 `ClinicalState` 作为共享状态，每个 Agent 读取前一步结果并写入自己的输出。
- 用 LangGraph 编排执行顺序，并在 Diagnosis 后支持条件路由。
- 服务层提供可独立测试的能力，例如 ICD-10 查询、药物相互作用、PHI 脱敏、FHIR 转换。

---

## 2. 准备环境

最低要求：

```text
Python 3.11+
Docker Desktop，可选但推荐
OpenAI API Key
Git，可选
```

Windows PowerShell 示例：

```powershell
python --version
docker --version
docker compose version
```

如果 Python 版本低于 3.11，先安装新版 Python，并确保安装时勾选 `Add Python to PATH`。

---

## 3. 新建项目目录

建议新建一个干净目录，不直接在原项目里改：

```powershell
cd E:\Projects
mkdir clinical-agent-python-rebuild
cd clinical-agent-python-rebuild
```

创建推荐目录结构：

```text
clinical-agent-python-rebuild/
  .env.example
  requirements.txt
  Dockerfile
  docker-compose.yml
  docker/
    init-db.sql
  data/
    sample_patients.json
  src/
    __init__.py
    api/
      __init__.py
      main.py
      routes.py
    agents/
      __init__.py
      intake_agent.py
      diagnosis_agent.py
      treatment_agent.py
      coding_agent.py
      audit_agent.py
    config/
      __init__.py
      settings.py
    graph/
      __init__.py
      state.py
      clinical_pipeline.py
    models/
      __init__.py
      patient.py
      diagnosis.py
      treatment.py
    services/
      __init__.py
      icd10_service.py
      drug_interaction.py
      graphrag_service.py
      fhir_service.py
      hipaa_service.py
  tests/
    __init__.py
    test_services.py
```

复现顺序不要按目录从上到下写，推荐按依赖关系写：

```text
配置 -> 数据模型 -> 服务层 -> Agent -> LangGraph Pipeline -> API -> 测试 -> Docker
```

---

## 4. 安装依赖

创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

`requirements.txt` 建议包含：

```text
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.0.0
httpx>=0.27.0
neo4j>=5.25.0
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.35
redis>=5.2.0
python-dotenv>=1.0.1
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0
fhir.resources>=7.1.0
numpy>=1.26.0
structlog>=24.4.0
tenacity>=9.0.0
pytest>=8.3.0
pytest-asyncio>=0.24.0
```

安装：

```powershell
pip install -r requirements.txt
```

---

## 5. 配置环境变量

创建 `.env.example`：

```text
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=clinical_decision
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass

REDIS_HOST=localhost
REDIS_PORT=6379

FHIR_SERVER_URL=http://localhost:8080/fhir

APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
```

复制一份真实配置：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入真实 `OPENAI_API_KEY`。

在 `src/config/settings.py` 中用 `pydantic-settings` 读取这些配置。重点字段包括：

- `openai_api_key`
- `openai_model`
- `postgres_dsn`
- `neo4j_uri`
- `fhir_server_url`
- `app_port`

---

## 6. 定义数据模型

先写 `models/`，因为 Agent 和 API 都依赖这些模型。

### 6.1 Patient 模型

在 `src/models/patient.py` 中定义：

- `Symptom`
- `Allergy`
- `Medication`
- `VitalSigns`
- `LabResult`
- `PatientInfo`

字段建议覆盖：

```text
name
age
gender
chief_complaint
symptoms
medical_history
family_history
allergies
current_medications
vital_signs
lab_results
```

### 6.2 Diagnosis 模型

在 `src/models/diagnosis.py` 中定义：

- `DiagnosisCandidate`
- `DifferentialDiagnosis`

核心字段：

```text
disease_name
icd10_hint
confidence
evidence
reasoning
recommended_tests
clinical_notes
knowledge_sources
```

### 6.3 Treatment / Coding / Audit 模型

在 `src/models/treatment.py` 中定义：

- `PrescribedMedication`
- `DrugInteraction`
- `TreatmentPlan`
- `ICD10Code`
- `DRGGroup`
- `CodingResult`
- `ComplianceCheck`
- `AuditRecord`
- `AuditResult`

这些模型不需要一开始就极度复杂，先保证能支撑 API 返回即可。

---

## 7. 定义共享状态 ClinicalState

在 `src/graph/state.py` 中定义 `ClinicalState`。

它是整个系统的核心数据结构：

```text
raw_input: 原始患者描述
patient_info: Intake Agent 输出
diagnosis: Diagnosis Agent 输出
needs_more_info: Diagnosis 后的条件路由标志
treatment_plan: Treatment Agent 输出
coding_result: Coding Agent 输出
audit_result: Audit Agent 输出
messages: LangGraph/LangChain 消息历史
errors: 流程中收集的错误
current_agent: 当前执行到哪个 Agent
```

设计原则：

- 不要让 Agent 之间直接互相调用。
- Agent 只读写 `ClinicalState`。
- 每个 Agent 返回 `dict`，由 LangGraph 合并进 state。
- 出错时不要直接中断全部流程，优先把错误写入 `errors`。

---

## 8. 先实现服务层

服务层要尽量不依赖 LLM，这样可以单独测试。

### 8.1 ICD-10 Service

文件：`src/services/icd10_service.py`

实现能力：

- `lookup_icd10(code)`
- `search_icd10_by_text(text)`
- `get_drg_group(icd10_code)`
- `validate_icd10_code(code)`

第一版可以用内置字典，例如：

```text
J18.9 -> Pneumonia, unspecified organism
I10 -> Essential hypertension
E11.9 -> Type 2 diabetes mellitus without complications
```

### 8.2 Drug Interaction Service

文件：`src/services/drug_interaction.py`

实现能力：

- `check_interactions(new_drugs, current_drugs)`
- `check_allergy_contraindication(drug, allergies)`

第一版内置常见交互：

```text
warfarin + aspirin -> major
ssri + maoi -> contraindicated
metformin + contrast_dye -> major
ace_inhibitor + potassium_supplement -> moderate
```

### 8.3 GraphRAG Service

文件：`src/services/graphrag_service.py`

第一版不要急着接 Neo4j，先做离线 map：

```text
fever -> Influenza, Pneumonia, COVID-19, Sepsis
cough -> Pneumonia, Bronchitis, Asthma, COPD
chest_pain -> Acute MI, Angina, Pulmonary Embolism
```

实现：

- `find_diseases_by_symptoms(symptoms)`
- `get_icd10(disease_name)`
- 预留 `query_neo4j(cypher, params)`

### 8.4 FHIR Service

文件：`src/services/fhir_service.py`

实现：

- `patient_to_fhir(patient_info)`
- `diagnosis_to_fhir_condition(diagnosis, patient_id)`
- `medication_to_fhir(medication, patient_id)`
- `push_to_fhir_server(resource)`

第一版只做 JSON 转换，FHIR Server 推送可以失败后返回 `None`。

### 8.5 HIPAA Service

文件：`src/services/hipaa_service.py`

实现：

- `detect_phi(text)`
- `deidentify_text(text)`
- `hash_identifier(value)`
- `AuditLogger`

第一版用正则覆盖：

```text
name
date
phone
email
ssn
mrn
url
ip
address
unique id
```

---

## 9. 实现 5 个 Agent

建议每个 Agent 都遵守同一套模式：

```text
读取 state 中的必要字段
检查输入是否存在
构造 System Prompt + Human Prompt
调用 ChatOpenAI
解析 JSON
用 Pydantic 模型校验
返回要写入 state 的 dict
捕获异常并写入 errors
```

### 9.1 Intake Agent

文件：`src/agents/intake_agent.py`

输入：

```text
state.raw_input
```

输出：

```text
patient_info
current_agent = "intake"
```

目标：

- 从患者自然语言描述中抽取姓名、年龄、性别、主诉。
- 抽取症状、既往史、过敏史、当前用药、生命体征、实验室检查。
- 返回严格 JSON。

### 9.2 Diagnosis Agent

文件：`src/agents/diagnosis_agent.py`

输入：

```text
state.patient_info
```

输出：

```text
diagnosis
needs_more_info
current_agent = "diagnosis"
```

目标：

- 生成主诊断。
- 生成 2 到 3 个鉴别诊断。
- 给出证据链和置信度。
- 如果信息不足，设置 `needs_more_info = true`。

### 9.3 Treatment Agent

文件：`src/agents/treatment_agent.py`

输入：

```text
state.patient_info
state.diagnosis
```

输出：

```text
treatment_plan
current_agent = "treatment"
```

目标：

- 根据诊断生成药物和非药物治疗建议。
- 检查当前用药和新药之间的风险。
- 检查过敏禁忌。
- 生成随访计划和 warning。

### 9.4 Coding Agent

文件：`src/agents/coding_agent.py`

输入：

```text
state.diagnosis
state.treatment_plan
```

输出：

```text
coding_result
current_agent = "coding"
```

目标：

- 生成主 ICD-10 编码。
- 生成次要 ICD-10 编码。
- 生成 DRG 分组。
- 给出编码理由和置信度。

### 9.5 Audit Agent

文件：`src/agents/audit_agent.py`

输入：

```text
patient_info
diagnosis
treatment_plan
coding_result
```

输出：

```text
audit_result
current_agent = "audit"
```

目标：

- 扫描 PHI。
- 生成合规检查项。
- 记录审计轨迹。
- 给出整体风险等级和建议。

Audit Agent 可以先用规则实现，不需要调用 LLM。

---

## 10. 实现 LangGraph Pipeline

文件：`src/graph/clinical_pipeline.py`

核心逻辑：

```text
创建 StateGraph(ClinicalState)
注册 intake / diagnosis / treatment / coding / audit 五个节点
设置入口为 intake
添加 intake -> diagnosis
添加 diagnosis 条件边：
  needs_more_info == true -> intake
  needs_more_info == false -> treatment
添加 treatment -> coding
添加 coding -> audit
添加 audit -> END
compile 并返回 pipeline
```

注意事项：

- 加 `MemorySaver` 作为 checkpointer。
- 暴露 `build_clinical_pipeline()`。
- 暴露单例 `get_pipeline()`，避免每次请求都重复构建图。

第一版可以先不做最大重试次数，但建议后续补上，避免 `needs_more_info` 一直为 true 造成循环。

---

## 11. 实现 FastAPI 接口

### 11.1 main.py

文件：`src/api/main.py`

职责：

- 创建 FastAPI app。
- 添加 CORS。
- 注册 router，prefix 使用 `/api/v1`。
- 暴露 `/health`。

### 11.2 routes.py

文件：`src/api/routes.py`

定义请求响应模型：

```text
AnalyzeRequest:
  patient_description: str
  thread_id: str = "default"

AnalyzeResponse:
  patient_info
  diagnosis
  treatment_plan
  coding_result
  audit_result
  errors
```

实现接口：

```text
POST /api/v1/clinical/analyze
POST /api/v1/clinical/icd10/search
GET  /api/v1/clinical/icd10/{code}
POST /api/v1/clinical/ddi/check
GET  /health
```

`/clinical/analyze` 的处理流程：

```text
pipeline = get_pipeline()
result = pipeline.invoke(
  {"raw_input": req.patient_description},
  config={"configurable": {"thread_id": req.thread_id}}
)
把 result 中的关键字段返回给客户端
```

---

## 12. 本地启动

进入项目根目录：

```powershell
cd E:\Projects\clinical-agent-python-rebuild
.\.venv\Scripts\Activate.ps1
```

启动：

```powershell
uvicorn src.api.main:app --reload --port 8000
```

访问：

```text
http://localhost:8000/health
http://localhost:8000/docs
```

测试请求：

```powershell
curl -X POST "http://localhost:8000/api/v1/clinical/analyze" `
  -H "Content-Type: application/json" `
  -d "{\"patient_description\":\"45-year-old male presenting with fever 39.2C for 3 days, productive cough with yellow sputum, right-sided chest pain. History of type 2 diabetes and hypertension. Current medications: metformin 500mg BID, lisinopril 10mg daily. Allergies: penicillin rash. Labs: WBC 15000, CRP 85, chest X-ray shows right lower lobe infiltrate.\"}"
```

---

## 13. 编写测试

先测服务层，因为不需要 OpenAI Key。

文件：`tests/test_services.py`

建议测试：

```text
lookup_icd10("J18.9") 能查到 Pneumonia
lookup_icd10("Z99.99") 返回 None
search_icd10_by_text("pneumonia") 返回 J18 相关编码
get_drg_group("J18.9") 返回 DRG 193
check_interactions(["warfarin"], ["aspirin"]) 返回 major
check_interactions(["metformin"], ["lisinopril"]) 返回空
detect_phi("Patient SSN is 123-45-6789") 能检测 ssn
deidentify_text 能移除 SSN、电话、邮箱
GraphRAGService.find_diseases_by_symptoms(["fever", "cough"]) 能召回 Pneumonia
```

运行：

```powershell
pytest tests/ -v
```

---

## 14. Docker 化

### 14.1 数据库初始化

文件：`docker/init-db.sql`

创建两张表：

```text
audit_logs
clinical_sessions
```

用途：

- `audit_logs`：保存审计记录。
- `clinical_sessions`：保存一次 Pipeline 的输入和输出。

第一版应用可以不立刻写数据库，但表结构先准备好。

### 14.2 Dockerfile

目标：

- 使用 Python 3.11 slim 镜像。
- 安装 `requirements.txt`。
- 拷贝 `src/`。
- 启动 `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`。

### 14.3 docker-compose.yml

服务：

```text
api: FastAPI 应用
postgres: PostgreSQL 16
neo4j: Neo4j 5 community
redis: Redis 7
```

端口：

```text
FastAPI: 8000
PostgreSQL: 5432
Neo4j Browser: 7474
Neo4j Bolt: 7687
Redis: 6379
```

启动：

```powershell
docker compose up --build
```

---

## 15. 推荐复现里程碑

### 里程碑 1：最小可运行 API

完成：

- FastAPI
- `/health`
- `/clinical/analyze`
- 假 Agent 返回 mock 数据

验收：

```text
curl /health 正常
Swagger 能打开
analyze 能返回五段结构
```

### 里程碑 2：真实 LangGraph Pipeline

完成：

- `ClinicalState`
- 5 个节点
- diagnosis 条件路由

验收：

```text
输入 raw_input 后，state 按顺序产生 patient_info、diagnosis、treatment_plan、coding_result、audit_result
```

### 里程碑 3：LLM Agent 可用

完成：

- Intake / Diagnosis / Treatment / Coding 调用 OpenAI。
- 每个 Agent 都要求模型返回纯 JSON。
- JSON 解析失败时写入 errors。

验收：

```text
完整病例输入后，返回结构化报告
```

### 里程碑 4：服务层完善

完成：

- ICD-10 查询
- DDI 检查
- PHI 检测和脱敏
- FHIR JSON 转换
- GraphRAG 离线召回

验收：

```text
pytest tests/ -v 通过
```

### 里程碑 5：Docker 一键启动

完成：

- Dockerfile
- docker-compose.yml
- PostgreSQL 初始化 SQL

验收：

```text
docker compose up --build
http://localhost:8000/docs 可访问
```

---

## 16. 容易踩坑的地方

### 16.1 LLM 返回不是合法 JSON

解决：

- Prompt 中明确写 `Return ONLY valid JSON, no markdown fences.`
- 如果返回 ```json 代码块，先剥掉 fence。
- 捕获 `json.JSONDecodeError`，写入 `errors`。

### 16.2 LangGraph 状态合并问题

解决：

- Agent 返回 dict，而不是直接改 state。
- `errors` 如果需要追加，要注意 reducer 或手动 `state.errors + [new_error]`。
- `messages` 使用 `add_messages`。

### 16.3 Diagnosis 回环可能无限循环

解决：

- 第一版可以先不触发回环。
- 后续给 `ClinicalState` 增加 retry counter。
- 或者在 Pipeline 中设置 recursion limit。

### 16.4 OpenAI Key 未配置

表现：

```text
Intake / Diagnosis / Treatment / Coding 全部报错
```

解决：

- 检查 `.env` 是否在项目根目录。
- 检查变量名是否为 `OPENAI_API_KEY`。
- 检查 `Settings.Config.env_file = ".env"`。

### 16.5 服务层和 Agent 脱节

当前原项目里部分 Agent 主要依赖 LLM，服务层更多是演示和辅助接口。复现时推荐逐步增强：

- Diagnosis Agent 可以调用 GraphRAG Service，把候选疾病加入 prompt。
- Treatment Agent 可以调用 DDI Service，对 LLM 推荐药物做二次校验。
- Coding Agent 可以调用 ICD-10 Service，校验 LLM 生成的 code 是否存在。
- Audit Agent 可以复用 HIPAA Service，避免重复写 PHI 正则。

---

## 17. 从演示版升级到更像生产版

可以按以下顺序增强：

```text
1. 给每个 Agent 增加输入输出 schema 强校验
2. 给 Diagnosis 回环增加最大重试次数
3. 把 Pipeline 运行结果持久化到 PostgreSQL
4. 把 AuditLogger 写入 audit_logs 表
5. 把 GraphRAG 从内置字典迁移到 Neo4j
6. 引入 Redis 做限流和缓存
7. 增加鉴权、RBAC、最小必要访问控制
8. 增加 OpenTelemetry / structlog 请求链路日志
9. 增加真实医疗编码库和药物库
10. 增加前端或 Swagger 示例集合
```

---

## 18. 推荐学习顺序

如果你是为了面试或自己真正吃透这个项目，建议按这个顺序讲给自己听：

```text
1. 为什么医疗场景适合 Pipeline，而不是 Supervisor 或 Debate
2. ClinicalState 为什么是多 Agent 系统的核心
3. 5 个 Agent 各自解决什么问题
4. Diagnosis 后为什么需要条件路由
5. 哪些能力交给 LLM，哪些能力放在服务层
6. 如何处理 LLM JSON 不稳定
7. 如何做医疗合规审计
8. 如何从 demo 演进到生产级系统
```

能把这 8 点讲清楚，基本就真正理解这个项目了。

