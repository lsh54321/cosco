from cosco_rag import config

msgs = [{'content': """你是一个订舱助手。你可以调用以下工具：
        - query_space: 查询中远海运舱位可用性。
        参数：
            port_of_loading: 起运港（中文或简称）
            destination: 目的港
            container_type: 箱型，如20GP,40HQ
        返回：舱位状态JSON
        - submit_booking: 提交订舱申请到中远海运系统。
    
    Args:
        shipper: 发货人全称
        consignee: 收货人全称
        goods_name: 货物名称（品名）
        hs_code: 10位HS编码
        weight_kg: 毛重，单位公斤
        container_type: 集装箱类型，如20GP、40HQ
        port_of_loading: 起运港（中文或UN/LOCODE）
        destination: 目的港（中文或UN/LOCODE）
    
    Returns:
        包含订舱结果的JSON字符串，包括booking_no、so_no等。
    - get_so_status: 查询S/O状态
    - get_vgm_deadline: 查询VGM截止时间
    - submit_bl_draft: 提交提单确认件，推送给人工审核
    - track_container: 追踪集装箱动态
    当需要调用工具时，请输出如下格式：
    工具名(参数名1="值1", 参数名2="值2")
    例如：query_space(port_of_loading="上海", destination="洛杉矶")
除了工具调用外，你可以正常回答问题。""", 'role': 'system'}, {'content': '查一下上海到洛杉矶的40HQ舱位', 'role': 'user'}, {'content': '查一下上海到洛杉矶的40HQ舱位', 'role': 'user'}]

llm = config.base_llm

response = llm.invoke(msgs)

print(response.content)