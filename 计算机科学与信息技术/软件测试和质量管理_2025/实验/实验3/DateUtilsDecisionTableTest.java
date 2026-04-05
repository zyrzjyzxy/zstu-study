package Test;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import java.util.Arrays;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class DateUtilsDecisionTableTest {
    private int year;
    private int month;
    private int day;
    private String expected;

    public DateUtilsDecisionTableTest(int year, int month, int day, String expected) {
        this.year = year;
        this.month = month;
        this.day = day;
        this.expected = expected;
    }

    // 定义测试数据（判定表驱动法）
    @Parameters(name = "{index}: {0}-{1}-{2} → {3}")
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][] {
                {2000, 5, 10, "2000-05-11"},  // 规则1：非最后一天
                {2000, 4, 30, "2000-05-01"},  // 规则2：非12月最后一天 → 跨月
                {2000, 12, 31, "2001-01-01"}, // 规则3：12月最后一天 → 跨年
                {2000, 2, 29, "2000-03-01"},  // 规则4：闰年2月结束
                {1900, 2, 28, "1900-03-01"}   // 规则5：非闰年2月结束
        });
    }

    @Test
    public void testDecisionTable() {
        String actual = DateUtils.getNextDate(year, month, day);
        assertEquals(expected, actual);
    }
}