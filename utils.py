def get_time_str_from_sec(total_sec: int):
    # From https://stackoverflow.com/a/539360
    days, remainder = divmod(total_sec, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{int(days)}d:{int(hours)}h:{int(minutes)}m:{int(seconds):02d}s"
    elif hours > 0:
        return f"{int(hours)}h:{int(minutes)}m:{int(seconds):02d}s"
    else:
        return f"{int(minutes)}m:{int(seconds):02d}s"
