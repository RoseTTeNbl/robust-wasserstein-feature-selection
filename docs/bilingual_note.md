# 理论稿维护

默认只维护英文规范来源 `note/main.tex`，`note/main.md` 是其完整阅读副本。
保留已有 `note/main_zh.tex` 和中文 PDF；英文更新不再要求同步中文。
实验报告的双语维护规则保持不变。

完成英文全文、公式和参考文献审校后，记录指纹并构建：

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by reviewer --review-note "Full English review"
.venv-research/Scripts/python.exe scripts/check_note_sync.py
./note/build.ps1
```

默认只读取英文源码及 `note/references.bib`，审校记录保存在 `note/english_review.json`。
中文源码或双语审校记录缺失、过时均不影响英文流程；构建仅替换英文 PDF。

用户要求双语交付时，先完整同步中文正文、定义、公式、命题、证明、算法和附录，
保持标签、结构、数学表达、参考文献键及引用顺序对应，再运行：

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --bilingual --record --reviewed-by reviewer --review-note "Full bilingual review"
.venv-research/Scripts/python.exe scripts/check_note_sync.py --bilingual
./note/build.ps1 -Bilingual
```

双语审校记录独立保存在 `note/bilingual_sync.json`，只有两版都编译通过才一起发布。
显式双语审校不替代英文记录；英文打包前仍须有当前的英文审校记录。

检查器校验已审校来源的指纹和结构，双语模式还比较两版的完整数学结构；
这些检查不能代替语义审校或 PDF 验证。摘要可省略，但不能出现多个摘要。
构建在独占 TEMP 中保存所选源码及参考文献的不可变副本，按
`pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX` 编译；
只有全部选定版本成功、引用和排版诊断通过，且来源与审校记录未变化时，
才替换对应正式 PDF，发布失败则恢复旧版。交付前检查本次构建的全部页面。

英文使用 newtx 字体；中文使用 CJKutf8 与 gbsn 字体，两版均使用 pdfLaTeX。
所需字体随常规 TeX 发行版提供，不依赖 Windows 专有字体。

在线理论包另用 `scripts/package_overleaf.py --verify` 生成，直接打包已审校的英文源码及
参考文献，不读取中文或双语记录，不改写前言或正文；本地成功不表示 Overleaf 已完成更新。
