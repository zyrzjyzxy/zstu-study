package net.mooctest;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * 合并CalculatorTest、MathParserTest、ExpressionParserTest为一个文件
 */
public class AllTest {

    // CalculatorTest 部分
    @Test
    public void testCommandNotFoundException() {
        CommandNotFoundException e = new CommandNotFoundException("cmd");
        assertEquals("cmd", e.getCommand());
    }

    @Test
    public void testExpressionParserException() {
        ExpressionParserException e = new ExpressionParserException("expr");
        assertEquals("expr", e.getFaultyExpression());
    }

    @Test
    public void testFunctionNotFoundException() {
        FunctionNotFoundException e1 = new FunctionNotFoundException("func");
        assertEquals("func", e1.getFunc());
        FunctionNotFoundException e2 = new FunctionNotFoundException("expr", "func2");
        assertEquals("func2", e2.getFunc());
    }

    @Test
    public void testMissingOperandException() {
        MissingOperandException e = new MissingOperandException("expr", "+");
        assertEquals("+", e.getOperator());
    }

    @Test
    public void testUnmatchedBracketsException() {
        UnmatchedBracketsException e = new UnmatchedBracketsException("expr", 5);
        assertEquals(5, e.getIndexOfBracket());
    }

    @Test
    public void testVariableNotFoundException() {
        VariableNotFoundException e1 = new VariableNotFoundException("var");
        assertEquals("var", e1.getVar());
        VariableNotFoundException e2 = new VariableNotFoundException("expr", "var2");
        assertEquals("var2", e2.getVar());
    }

    @Test(expected = NullExpressionException.class)
    public void testEvaluateNullExpression() throws Exception {
        Calculator.expParser = new ExpressionParser(2);
        Calculator.evaluate("");
    }

    @Test
    public void testEvaluateNumberAndAssignment() throws Exception {
        Calculator.expParser = new ExpressionParser(2);
        Calculator.previousAns = "0";
        assertEquals("8.0", Calculator.evaluate("8"));
        assertEquals("8.0", Calculator.evaluate(" 8 = 8"));
    }

    @Test(expected = ExpressionParserException.class)
    public void testEvaluateInvalidAssignment() throws Exception {
        Calculator.expParser = new ExpressionParser(2);
        Calculator.evaluate("1+2=3");
    }

    @Test
    public void testAddVariable() throws Exception {
        Calculator.expParser = new ExpressionParser(2);
        Calculator.evaluate("a = 5");
        Calculator.expParser.addVariable("a", "10");
        assertEquals("10", Calculator.expParser.variables[0][1]);
    }

    @Test
    public void testParseCommandHelpAndList() throws Exception {
        Calculator.expParser = new ExpressionParser(2);
        Calculator.previousAns = "0";
        Calculator.parseCommand("exit");
        Calculator.parseCommand("help");
        Calculator.parseCommand("help vars");
        Calculator.parseCommand("help funcs");
        Calculator.parseCommand("help cmds");
        Calculator.parseCommand("list");
        Calculator.parseCommand("list cmds");
        Calculator.parseCommand("list funcs");
    }

    @Test(expected = CommandNotFoundException.class)
    public void testParseCommandUnknown() throws Exception {
        Calculator.parseCommand("unknown");
    }

    @Test
    public void testMathParserFactorialAndBinaryOperation() {
        assertEquals(1.0, MathParser.factorial(1), 0.0001);
        assertEquals(2.0, MathParser.factorial(2), 0.0001);
        assertEquals(16.0, MathParser.solveBinaryOperation(4, "^", 2), 0.0001);
        assertEquals(0.0, MathParser.solveBinaryOperation(4, "%", 2), 0.0001);
        assertEquals(2.0, MathParser.solveBinaryOperation(4, "/", 2), 0.0001);
        assertEquals(8.0, MathParser.solveBinaryOperation(4, "*", 2), 0.0001);
        assertEquals(6.0, MathParser.solveBinaryOperation(4, "+", 2), 0.0001);
        assertEquals(2.0, MathParser.solveBinaryOperation(4, "-", 2), 0.0001);
        assertEquals(0.0, MathParser.solveBinaryOperation(4, "--", 2), 0.0001);
    }

    @Test
    public void testMathParserSolveUnaryFunction() throws Exception {
        assertEquals(Math.sin(30), MathParser.solveUnaryFunction("sin", 30), 0.0001);
        assertEquals(Math.cos(30), MathParser.solveUnaryFunction("cos", 30), 0.0001);
        assertEquals(Math.tan(30), MathParser.solveUnaryFunction("tan", 30), 0.0001);
        assertEquals(1.0 / Math.sin(30), MathParser.solveUnaryFunction("csc", 30), 0.0001);
        assertEquals(1.0 / Math.cos(30), MathParser.solveUnaryFunction("sec", 30), 0.0001);
        assertEquals(1.0 / Math.tan(30), MathParser.solveUnaryFunction("ctn", 30), 0.0001);
        assertEquals(Math.toRadians(30), MathParser.solveUnaryFunction("rad", 30), 0.0001);
        assertEquals(Math.toDegrees(30), MathParser.solveUnaryFunction("deg", 30), 0.0001);
        assertEquals(MathParser.factorial(5), MathParser.solveUnaryFunction("fct", 5), 0.0001);
        assertEquals(Math.abs(-30), MathParser.solveUnaryFunction("abs", -30), 0.0001);
        assertEquals(Math.exp(5), MathParser.solveUnaryFunction("exp", 5), 0.0001);
        assertEquals(Math.log(30), MathParser.solveUnaryFunction("log", 30), 0.0001);
    }

    @Test(expected = FunctionNotFoundException.class)
    public void testMathParserSolveUnaryFunctionUnknown() throws Exception {
        MathParser.solveUnaryFunction("unknown", 1);
    }

    @Test
    public void testParseVariables() throws Exception {
        ExpressionParser parser = new ExpressionParser(2);
        parser.addVariable("x", "10");
        assertEquals("10 + 5", parser.parseVariables("<x> + 5").replaceAll("\\s+", " ").trim());
    }

    @Test(expected = VariableNotFoundException.class)
    public void testParseVariablesUnknown() throws Exception {
        ExpressionParser parser = new ExpressionParser(2);
        parser.parseVariables("<y> + 5");
    }

    @Test
    public void testParseParenthesis() throws Exception {
        ExpressionParser parser = new ExpressionParser(2);
        assertEquals("5.0", parser.parseParenthesis("(2 + 3)"));
        assertEquals("9.0", parser.parseParenthesis("(2 + (3 * 4) - 5)"));
    }

    @Test(expected = UnmatchedBracketsException.class)
    public void testParseParenthesisUnmatched() throws Exception {
        ExpressionParser parser = new ExpressionParser(2);
        parser.parseParenthesis("(2 + 3");
    }

    @Test
    public void testParseFunctions() throws Exception {
        ExpressionParser parser = new ExpressionParser(2);
        assertEquals("120.0", parser.parseFunctions("fct[5]"));
    }

    @Test(expected = FunctionNotFoundException.class)
    public void testParseFunctionsUnknown() throws Exception {
        ExpressionParser parser = new ExpressionParser(2);
        parser.parseFunctions("abc[5]");
    }

    @Test
    public void testParseOperators() throws Exception {
        ExpressionParser parser = new ExpressionParser(2);
        assertEquals("5.0", parser.parseOperators("2 + 3"));
        assertEquals("1.0", parser.parseOperators("3 - 2"));
    }

    @Test(expected = MissingOperandException.class)
    public void testParseOperatorsMissingOperand() throws Exception {
        ExpressionParser parser = new ExpressionParser(2);
        parser.parseOperators("3 +");
    }

    @Test
    public void testAdjustNumberSpacing() {
        String result = ExpressionParser.adjustNumberSpacing("2+3");
        assertTrue(result.contains("2"));
        assertTrue(result.contains("+3"));
    }

    @Test
    public void testIndexOfMatchingBracket() throws Exception {
        String str = "(2 + (3 * 4))";
        int index = ExpressionParser.indexOfMatchingBracket(str, 0, '(', ')');
        assertEquals(12, index);
    }

    @Test(expected = UnmatchedBracketsException.class)
    public void testIndexOfMatchingBracketUnmatched() throws Exception {
        String str = "(2 + (3 * 4)";
        ExpressionParser.indexOfMatchingBracket(str, 0, '(', ')');
    }

    // MathParserTest 部分
    @Test
    public void testIsNumber() {
        assertTrue(MathParser.isNumber("123"));
        assertTrue(MathParser.isNumber("123.456"));
        assertTrue(MathParser.isNumber("-123"));
        assertTrue(MathParser.isNumber("-123.456"));
        assertTrue(MathParser.isNumber("1.23e4"));
        assertTrue(MathParser.isNumber("1.23E-4"));
        assertFalse(MathParser.isNumber("abc"));
        assertFalse(MathParser.isNumber(""));
        assertFalse(MathParser.isNumber("12a"));
        assertFalse(MathParser.isNumber("12.34.56"));
        assertFalse(MathParser.isNumber("12..34"));
        assertFalse(MathParser.isNumber("12-34"));
    }

    @Test
    public void testFactorial() {
        assertEquals(1.0, MathParser.factorial(0), 0.0001);
        assertEquals(1.0, MathParser.factorial(1), 0.0001);
        assertEquals(2.0, MathParser.factorial(2), 0.0001);
        assertEquals(6.0, MathParser.factorial(3), 0.0001);
        assertEquals(24.0, MathParser.factorial(4), 0.0001);
        assertEquals(120.0, MathParser.factorial(5), 0.0001);
        assertEquals(720.0, MathParser.factorial(6), 0.0001);
        assertEquals(3628800.0, MathParser.factorial(10), 0.0001);
    }

    @Test
    public void testSolveBinaryOperation() {
        assertEquals(8.0, MathParser.solveBinaryOperation(2, "^", 3), 0.0001);
        assertEquals(0.5, MathParser.solveBinaryOperation(2, "^", -1), 0.0001);
        assertEquals(1.0, MathParser.solveBinaryOperation(5, "%", 2), 0.0001);
        assertEquals(0.0, MathParser.solveBinaryOperation(4, "%", 2), 0.0001);
        assertEquals(2.5, MathParser.solveBinaryOperation(5, "/", 2), 0.0001);
        assertEquals(Double.POSITIVE_INFINITY, MathParser.solveBinaryOperation(5, "/", 0), 0.0001);
        assertEquals(10.0, MathParser.solveBinaryOperation(5, "*", 2), 0.0001);
        assertEquals(-10.0, MathParser.solveBinaryOperation(-5, "*", 2), 0.0001);
        assertEquals(7.0, MathParser.solveBinaryOperation(5, "+", 2), 0.0001);
        assertEquals(3.0, MathParser.solveBinaryOperation(5, "+", -2), 0.0001);
        assertEquals(3.0, MathParser.solveBinaryOperation(5, "-", 2), 0.0001);
        assertEquals(7.0, MathParser.solveBinaryOperation(5, "-", -2), 0.0001);
        assertEquals(0.0, MathParser.solveBinaryOperation(5, "unknown", 2), 0.0001);
    }

    @Test
    public void testSolveUnaryFunction() throws FunctionNotFoundException {
        assertEquals(Math.sin(30), MathParser.solveUnaryFunction("sin", 30), 0.0001);
        assertEquals(Math.cos(30), MathParser.solveUnaryFunction("cos", 30), 0.0001);
        assertEquals(Math.tan(30), MathParser.solveUnaryFunction("tan", 30), 0.0001);
        assertEquals(1.0/Math.sin(30), MathParser.solveUnaryFunction("csc", 30), 0.0001);
        assertEquals(1.0/Math.cos(30), MathParser.solveUnaryFunction("sec", 30), 0.0001);
        assertEquals(1.0/Math.tan(30), MathParser.solveUnaryFunction("ctn", 30), 0.0001);
        assertEquals(Math.toRadians(30), MathParser.solveUnaryFunction("rad", 30), 0.0001);
        assertEquals(Math.toDegrees(30), MathParser.solveUnaryFunction("deg", 30), 0.0001);
        assertEquals(MathParser.factorial(5), MathParser.solveUnaryFunction("fct", 5), 0.0001);
        assertEquals(Math.abs(-30), MathParser.solveUnaryFunction("abs", -30), 0.0001);
        assertEquals(Math.exp(5), MathParser.solveUnaryFunction("exp", 5), 0.0001);
        assertEquals(Math.log(30), MathParser.solveUnaryFunction("log", 30), 0.0001);
    }

    @Test(expected = FunctionNotFoundException.class)
    public void testSolveUnaryFunctionException() throws FunctionNotFoundException {
        MathParser.solveUnaryFunction("unknown", 5);
    }

    // ExpressionParserTest 部分
    @Test
    public void testConstructor() {
        ExpressionParser parser = new ExpressionParser(10);
        assertNotNull(parser.variables);
        assertEquals(0, parser.numberOfVars);
    }

    @Test
    public void testAddVariable2() {
        ExpressionParser parser = new ExpressionParser(10);
        parser.addVariable("x", "10");
        assertEquals("10", parser.variables[0][1]);
        parser.addVariable("x", "20");
        assertEquals("20", parser.variables[0][1]);
    }

    @Test
    public void testEvaluateNumber() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("10.0", parser.evaluate("10"));
    }

    @Test
    public void testEvaluateAssignment() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("20.0", parser.evaluate("x = 20"));
        assertEquals("20.0", parser.variables[0][1]);
    }

    @Test
    public void testEvaluateExpression() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("5.0", parser.evaluate("2 + 3"));
    }

    @Test(expected = NullExpressionException.class)
    public void testEvaluateNullExpression2() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        parser.evaluate("");
    }

    @Test
    public void testEvaluateComplexExpression() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("14.0", parser.evaluate("2 + 3 * 4"));
        assertEquals("20.0", parser.evaluate("(2 + 3) * 4"));
    }

    @Test(expected = ExpressionParserException.class)
    public void testEvaluateInvalidExpression() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        parser.evaluate("2 + + 3");
    }

    @Test
    public void testParseVariables2() throws VariableNotFoundException {
        ExpressionParser parser = new ExpressionParser(10);
        parser.addVariable("x", "10");
        assertEquals("10 + 5", parser.parseVariables("<x> + 5").replaceAll("\\s+", " ").trim());
    }

    @Test(expected = VariableNotFoundException.class)
    public void testParseVariablesException() throws VariableNotFoundException {
        ExpressionParser parser = new ExpressionParser(10);
        parser.parseVariables("<y> + 5");
    }

    @Test
    public void testParseParenthesis2() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("5.0", parser.parseParenthesis("(2 + 3)"));
    }

    @Test
    public void testParseParenthesisNested() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("9.0", parser.parseParenthesis("(2 + (3 * 4) - 5)"));
    }

    @Test(expected = UnmatchedBracketsException.class)
    public void testParseParenthesisException() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        parser.parseParenthesis("(2 + 3");
    }

    @Test
    public void testParseFunctions2() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("120.0", parser.parseFunctions("fct[5]"));
    }

    @Test(expected = FunctionNotFoundException.class)
    public void testParseFunctionsUnknownFunction() throws ExpressionParserException {
        ExpressionParser parser = new ExpressionParser(10);
        parser.parseFunctions("unknown[5]");
    }

    @Test
    public void testParseOperators2() throws MissingOperandException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("5.0", parser.parseOperators("2 + 3"));
    }

    @Test
    public void testParseOperatorsWithSubtraction() throws MissingOperandException {
        ExpressionParser parser = new ExpressionParser(10);
        assertEquals("1.0", parser.parseOperators("3 - 2"));
    }

    @Test(expected = MissingOperandException.class)
    public void testParseOperatorsMissingOperand2() throws MissingOperandException {
        ExpressionParser parser = new ExpressionParser(10);
        parser.parseOperators("3 +");
    }

    @Test
    public void testAdjustNumberSpacing2() {
        String result = ExpressionParser.adjustNumberSpacing("2+3");
        assertEquals("2 + +3", result.trim());
    }

    @Test
    public void testAdjustNumberSpacingComplexExpression() {
        String result = ExpressionParser.adjustNumberSpacing("2+3*4-5");
        assertEquals("2 + +3 * 4 + -5", result.trim().replaceAll("\\s+", " "));
    }

    @Test
    public void testIndexOfMatchingBracket2() throws UnmatchedBracketsException {
        String str = "(2 + (3 * 4))";
        int index = ExpressionParser.indexOfMatchingBracket(str, 0, '(', ')');
        assertEquals(12, index);
    }

    @Test
    public void testIndexOfMatchingBracketNested() throws UnmatchedBracketsException {
        String str = "(2 + (3 * (4 - 1)))";
        int index = ExpressionParser.indexOfMatchingBracket(str, 0, '(', ')');
        assertEquals(18, index);
    }

    @Test(expected = UnmatchedBracketsException.class)
    public void testIndexOfMatchingBracketException() throws UnmatchedBracketsException {
        String str = "(2 + (3 * 4)";
        ExpressionParser.indexOfMatchingBracket(str, 0, '(', ')');
    }
}