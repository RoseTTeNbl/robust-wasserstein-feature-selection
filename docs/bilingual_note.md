# 双语理论稿维护

规范来源是 `note/main.tex` 和 `note/main_zh.tex`，`main.md` 是完整英文阅读副本。
修改须同步全部正文、定义、公式、命题、证明、算法和附录，保持标签及结构对应。
两版共用 `note/references.bib`，参考文献键及引用顺序保持对应。

完成语义和数学审校后，再记录指纹并成对构建：

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by reviewer --review-note "Full bilingual review"
.venv-research/Scripts/python.exe scripts/check_note_sync.py
./note/build.ps1
```

检查器只比较已审校来源的指纹和数学结构，不能代替语义审校或 PDF 验证。
构建在独占 TEMP 中保存每版源码及共享参考文献的不可变副本，按
`pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX` 编译；只有两份均成功、引用和排版检查通过，
且来源与审校记录未变化时，才一起替换正式 PDF。交付前检查两版全部页面。

英文使用 newtx 字体；中文使用 CJKutf8 与 gbsn 字体，两版均使用 pdfLaTeX。
所需字体随常规 TeX 发行版提供，不依赖 Windows 专有字体。

实验报告独立维护中英文来源。在线理论包另用 `scripts/package_overleaf.py --verify`
生成，直接打包已审校的英文源码及共享参考文献，不改写前言或正文；
本地成功不表示 Overleaf 已完成更新。
