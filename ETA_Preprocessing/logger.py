import logging


def get_logger():
    logger = logging.getLogger('DeepTravel')  # 创建一个日志对象，命名为DeepTravel
    # Handler.setLevel(lel): 指定被处理的信息级别，低于lel级别的信息将被忽略
    logger.setLevel(logging.DEBUG)  # 日志等级：DEBUG：最详细的日志信息，典型应用场景是 问题诊断

    # 创建一个handler：将(logger创建的)日志记录发送到合适的目的输出；
    ch = logging.StreamHandler()  # ():默认输出到sys.stderr,用于输出到控制台
    # 如果把logger的级别设置为INFO，那么小于INFO级别的日志都不输出，大于等于INFO级别的日志都输出
    ch.setLevel(logging.INFO)  # 日志等级：INFO：通常只记录关键节点信息，用于确认一切都是按照我们预期的那样进行工作
    formatter = logging.Formatter('%(message)s')  # 决定日志记录的最终输出格式；%(message)s：用户输出的消息
    ch.setFormatter(formatter)  # 设置用于在Handler实例（ch）如上创建日志消息的消息格式化对象

    # 给我们开始实例化的logger对象添加handler
    logger.addHandler(ch)

    return logger




