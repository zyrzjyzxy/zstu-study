package Test;

public class DateUtils {
    /**
     * 计算输入日期的下一天
     * @param year 年（1900-2050）
     * @param month 月（1-12）
     * @param day 日（1-31）
     * @return 下一天的日期字符串，格式为 "yyyy-MM-dd"
     */
    public static String getNextDate(int year, int month, int day) {
        int[] monthDays = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        boolean isLeapYear = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);

        // 处理闰年2月
        if (isLeapYear && month == 2) {
            monthDays[1] = 29;
        }

        // 检查是否为最后一天
        if (day >= monthDays[month - 1]) {
            day = 1;
            if (month == 12) {
                month = 1;
                year++;
            } else {
                month++;
            }
        } else {
            day++;
        }

        return String.format("%04d-%02d-%02d", year, month, day);
    }
}