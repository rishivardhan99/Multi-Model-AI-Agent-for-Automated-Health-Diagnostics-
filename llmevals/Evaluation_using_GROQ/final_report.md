# LLMEVALS Final Report
* mode: pairwise

## LLM (raw) Metrics

- **count**: 6
- **mean**: 80.0
- **median**: 80.0
- **min**: 80.0
- **max**: 80.0

## System-validated Metrics

- **total evaluated**: 6
- **validated_system_pass_count**: 6
- **validated_system_accuracy**: 100.00%
- **abstention_count**: 2
- **abstention_rate**: 33.33%

## High severity issues (sample)

- **clean_00002_1768312272417.model1_final**: {'type': 'hallucination', 'severity': 'high', 'message': 'Model3 introduces a new claim of Cardiovascular Risk not supported by Model2.'}
- **clean_00019_1768312272520.model1_final**: {'type': 'hallucination', 'severity': 'high', 'message': "Model3 introduces a claim of 'Cardiovascular_Risk' without sufficient evidence."}

## Individual results (summary)

### clean_00001_1768312272406.model1_final
- raw_overall_score: `80`
- validation: `{'raw_score': 80.0, 'penalty': 7.0, 'validated_score': 73.0, 'is_abstain': False, 'abstention_type': None, 'system_pass': True, 'reasons': ['validated_score_meets_soft_threshold']}`

### clean_00002_1768312272417.model1_final
- raw_overall_score: `80`
- validation: `{'raw_score': 80.0, 'penalty': 22.0, 'validated_score': 58.0, 'is_abstain': True, 'abstention_type': 'clinical', 'system_pass': True, 'reasons': ['clinical_abstention_due_to_high_risk_output']}`

### clean_00003_1768312272436.model1_final
- raw_overall_score: `80`
- validation: `{'raw_score': 80.0, 'penalty': 7.0, 'validated_score': 73.0, 'is_abstain': False, 'abstention_type': None, 'system_pass': True, 'reasons': ['validated_score_meets_soft_threshold']}`

### clean_00009_1768312272453.model1_final
- raw_overall_score: `(no score)`
- validation: `{'raw_score': None, 'penalty': 0.0, 'validated_score': 0.0, 'is_abstain': True, 'abstention_type': 'technical', 'system_pass': None, 'reasons': ['technical_abstention_llm_or_parse_error']}`

### clean_00011_1768312272473.model1_final
- raw_overall_score: `80`
- validation: `{'raw_score': 80.0, 'penalty': 14.0, 'validated_score': 66.0, 'is_abstain': False, 'abstention_type': None, 'system_pass': True, 'reasons': ['validated_score_meets_soft_threshold']}`

### clean_00012_1768312272490.model1_final
- raw_overall_score: `(no score)`
- validation: `{'raw_score': None, 'penalty': 0.0, 'validated_score': 0.0, 'is_abstain': True, 'abstention_type': 'technical', 'system_pass': None, 'reasons': ['technical_abstention_llm_or_parse_error']}`

### clean_00013_1768312272502.model1_final
- raw_overall_score: `(no score)`
- validation: `{'raw_score': None, 'penalty': 0.0, 'validated_score': 0.0, 'is_abstain': True, 'abstention_type': 'technical', 'system_pass': None, 'reasons': ['technical_abstention_llm_or_parse_error']}`

### clean_00019_1768312272520.model1_final
- raw_overall_score: `80`
- validation: `{'raw_score': 80.0, 'penalty': 22.0, 'validated_score': 58.0, 'is_abstain': True, 'abstention_type': 'clinical', 'system_pass': True, 'reasons': ['clinical_abstention_due_to_high_risk_output']}`

### clean_00020_1768312272531.model1_final
- raw_overall_score: `80`
- validation: `{'raw_score': 80.0, 'penalty': 7.0, 'validated_score': 73.0, 'is_abstain': False, 'abstention_type': None, 'system_pass': True, 'reasons': ['validated_score_meets_soft_threshold']}`

### clean_00021_1768312272547.model1_final
- raw_overall_score: `(no score)`
- validation: `{'raw_score': None, 'penalty': 0.0, 'validated_score': 0.0, 'is_abstain': True, 'abstention_type': 'technical', 'system_pass': None, 'reasons': ['technical_abstention_llm_or_parse_error']}`
