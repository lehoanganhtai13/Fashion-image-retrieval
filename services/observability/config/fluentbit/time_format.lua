function format_time(tag, timestamp, record)
    local time = record["time"]
    if time then
        local pattern = "(%d+-%d+-%d+ %d+:%d+:%d+)"
        local formatted_time = string.gsub(time, pattern, "%1"):gsub(" ", "T")
        record["time"] = formatted_time
    end
    return 1, timestamp, record
end