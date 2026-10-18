# 双语理论稿维护

规范来源是 `note/main.tex` 和 `note/main_zh.tex`，`main.md` 是完整英文阅读副本。
修改须同步全部正文、定义、公式、命题、证明、算法和附录，保持标签及结构对应。

完成语义和数学审校后，再记录指纹并成对构建：

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by reviewer --review-note "Full bilingual review"
.venv-research/Scripts/python.exe scripts/check_note_sync.py
./note/build.ps1
```

检查器只比较已审校来源的指纹和数学结构，不能代替语义审校或 PDF 验证。
构建在独占 TEMP 中编译两版，只有两份均成功才替换正式 PDF。交付前检查两版全部页面。

实验报告独立维护中英文来源。在线理论包另用 `scripts/package_overleaf.py --verify`
生成，开源字体转换只改前言，不改正文；本地成功不表示 Overleaf 已完成更新。
