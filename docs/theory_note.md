# 理论稿维护

`note/main.tex` 是规范英文源码，`note/main.md` 是其完整阅读副本。
只维护英文，第一节问题定义保持逐字不变。

完成全文、公式、证明、算法和参考文献审校后：

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by reviewer --review-note "Full English review"
.venv-research/Scripts/python.exe scripts/check_note_sync.py
./note/build.ps1
```

检查器验证源码与 `note/references.bib` 的审校指纹，记录保存在
`note/english_review.json`。它不能替代数学审查或 PDF 视觉检查。
构建在独占系统 TEMP 中保存源码快照，以 pdfLaTeX、BibTeX 和两次 pdfLaTeX 编译，
确认引用和排版诊断通过、来源未变化后，才替换正式英文 PDF。
交付前检查全部页面。字体使用 TeX 发行版提供的 newtx。

```powershell
.venv-research/Scripts/python.exe scripts/package_overleaf.py --verify --output note/robusthw_theory_overleaf.zip
```

Overleaf 包只包含当前英文源码、参考文献和编译说明；规范文件按字节复制。
该命令不会上传云端。具体操作见 [Overleaf 说明](overleaf.md)。
