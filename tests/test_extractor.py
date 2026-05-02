from anima.extractor import ExperienceExtractor, MockExtractor, ExtractionResult


class TestExtractionResult:
    def test_has_required_fields(self):
        result = ExtractionResult(
            concepts=["用户留存", "渠道分析"],
            entities=["SQL"],
            domain="data_analysis",
            problems=["数据缺失"],
            solutions=["中位数填充"],
            outcome_summary="成功找到原因",
            related_concepts=["用户流失"],
        )
        assert "用户留存" in result.concepts
        assert result.domain == "data_analysis"

    def test_default_values(self):
        result = ExtractionResult()
        assert result.concepts == []
        assert result.domain == "unknown"
        assert result.outcome_summary == ""


class TestMockExtractor:
    def test_returns_valid_result(self):
        extractor = MockExtractor()
        result = extractor.extract("分析用户留存数据，用SQL取数后按渠道分群")
        assert isinstance(result, ExtractionResult)
        assert len(result.concepts) > 0

    def test_detects_data_analysis_domain(self):
        extractor = MockExtractor()
        result = extractor.extract("分析用户留存数据趋势")
        assert result.domain == "data_analysis"

    def test_detects_code_writing_domain(self):
        extractor = MockExtractor()
        result = extractor.extract("编写一个Python脚本来处理代码")
        assert result.domain == "code_writing"

    def test_returns_unknown_for_ambiguous(self):
        extractor = MockExtractor()
        result = extractor.extract("帮我处理一下这个东西")
        assert result.domain == "unknown"

    def test_concepts_are_meaningful(self):
        extractor = MockExtractor()
        result = extractor.extract("分析用户留存数据")
        # Should have at least one concept longer than 1 char
        long_concepts = [c for c in result.concepts if len(c) >= 2]
        assert len(long_concepts) > 0
