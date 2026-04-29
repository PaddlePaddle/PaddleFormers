# PaddleFormers Bot

> 基于 Claude AI 的 GitHub 自动化 CI 分析机器人，在 PR 或 Issue 中 `@formers` 即可触发，自动分析 CI 失败原因并给出修复建议。

---

## 📋 目录

- [功能介绍](#功能介绍)
- [快速使用](#快速使用)

---

## 功能介绍

- 📥 自动下载 GitHub Actions CI 日志
- 🔍 使用 AI 分析失败原因
- 💬 将分析结果自动回复到 PR / Issue 评论区
- ⚡ 支持 PR 评论、Review、Issue 多种触发方式

---

## 快速使用

在任意 PR 或 Issue 的评论中 `@formers`，Bot 会自动触发分析：

```
@formers 帮我分析一下这个 PR 的 CI 失败原因
```

```
@formers 这个 Issue 的 CI 日志有什么问题？
```

注意 需要加上 PR 或者 Github Action 链接

Bot 会在几分钟内自动回复分析报告：

```
Formers Bot 分析结果

## CI 失败原因
...

## 修复建议
...

---
*由 @formers 自动生成*
```

### 触发方式

| 场景 | 触发条件 |
|------|---------|
| PR 评论 | 评论中包含 `@formers` |
| PR Review 评论 | Review 评论中包含 `@formers` |
| PR Review | Review 正文中包含 `@formers` |
| Issue | Issue 标题或正文中包含 `@formers` |
