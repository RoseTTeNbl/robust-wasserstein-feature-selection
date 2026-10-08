# 维护完整双语理论笔记

英文和中文是同一份持续维护的文稿。规范源文件为 `note/main.tex` 和 `note/main_zh.tex`；中文包含完整数学正文及所有附录。

修改后同步两版，并审校全文与公式。保留标签、引用、章节层级、附录位置、定理/证明和算法结构。标准库检查器还比较数学表达式，忽略空白及 `\text{subject to}` 等自然语言注释的翻译差异。显示公式严格保持顺序；行内表达允许同一段落内的中文语序调整，但不能增删、改写或跨段移动，算法按步骤核对。

完成审校后显式记录：

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by "reviewer" --review-note "Complete bilingual semantic and formula review"
.venv-research/Scripts/python.exe scripts/check_note_sync.py
./note/build.ps1
```

署名表示已经完成审校，而不是自动翻译结果。检查器核对指纹和完整结构，不能证明中文表达与英文语义等价。`note/bilingual_sync.json` 保存两源文件 SHA、结构指纹、计数、审校人及时间。英文改了而中文未改，禁止重新记录；任一版未经审校的改动都会阻止默认构建。

构建器将已审校源码复制到独占系统 TEMP，使用已有 XeLaTeX 各编译两次。编译期间源码或审校记录变化会使构建失效。两份成功后才替换正式 PDF；发布复制失败则恢复旧文件。失败时保留文件仍是旧构建，不标为当前译文。成功构建后仍需检查两版渲染页面。

检查器不需要第三方 Python 包，不会翻译文本。构建优先使用隔离研究环境的 Python，否则使用已有系统 Python。两版应包含完整独立正文；外部 TeX 输入需先明确扩展同步范围。实验报告不属于此翻译要求。

中文源采用宋体正文、黑体标题、Times New Roman 拉丁术语与 Cambria Math 数学字体。原生 XeTeX 中文断行配合 `fontspec`，避开本机较新 `xeCJK` 与较旧 LaTeX 内核的版本冲突；不依赖额外中文宏包。`\LatinWord{...}` 只控制拉丁术语字体，不改变数学含义。
