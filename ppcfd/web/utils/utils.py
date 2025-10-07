def detect_and_convert_units(value, current_unit, target_unit):
    """
    检测并转换单位。
    :param value: 数值
    :param current_unit: 当前单位（如 'm', 'km', 'cm'）
    :param target_unit: 目标单位（如 'm', 'km', 'cm'）
    :return: 转换后的数值
    """
    unit_conversion = {
        'm': 1,
        'km': 1000,
        'cm': 0.01,
        'mm':0.001
    }
    
    if current_unit not in unit_conversion or target_unit not in unit_conversion:
        raise ValueError(f"Unsupported unit: {current_unit} or {target_unit}")
    
    value_in_meters = value * unit_conversion[current_unit]
    converted_value = value_in_meters / unit_conversion[target_unit]
    
    return converted_value


def get_bounds_info(data):
    """
    获取数据的边界信息。
    :param data: 数值列表
    :return: 最小值和最大值
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    min_value = min(data)
    max_value = max(data)
    
    return min_value, max_value