# Engineering Agent

Engineering Agent 是一个面向软件工程任务的 AI Agent。

它区别于普通聊天 Agent，
不仅回答代码问题，还可以进入真实代码仓库，
完成代码调查、修改和验证。

## Project File Tools

Engineering Agent 提供项目文件操作能力。

read_project_file:
用于读取项目中的文件内容。

write_project_file:
用于修改项目文件。

list_project_files:
用于查看项目目录结构。

## Validation

代码修改完成后，
Engineering Agent 可以执行项目验证。

run_project_validation:
用于运行预定义的验证命令，
确认代码修改是否符合预期。

验证失败时，
Agent 可以根据结果继续分析问题。

## Conversation Continuation

当页面刷新导致 Agent 执行状态中断时，
系统可以恢复未完成的执行流程。

needsContinuation:
用于判断当前会话是否存在需要恢复的执行状态。

resumeContinuation:
用于恢复页面刷新前尚未完成的 Agent 执行流程。
