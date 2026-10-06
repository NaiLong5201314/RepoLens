import json
import time
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage
import main as backend


BASE_DIRECTORY = Path(__file__).resolve().parent

CASES_PATH = (
    BASE_DIRECTORY/ "evaluation_cases.json"
)

RESULTS_PATH = (
     BASE_DIRECTORY
    / "evaluation_results.json"
)

REFUSAL_MARKERS = [
    "not implemented",
    "not found",
    "could not find",
    "cannot find",
    "does not contain",
    "doesn't contain",
    "not implemented",
    "no evidence",
    "no authentication",
    "no login",
    "there is no",
    "没有找到",
    "不存在",
    "未实现",
]

def load_cases() -> list[dict]:
    return json.loads(
        CASES_PATH.read_text(
            encoding="utf-8"
        )
    )

def check_sources(
        expected_sources : list[str],
        actual_sources: list[str],
) -> bool:
    if not expected_sources:
        return True


    normalized_actual = [
        source.lower()
        for source in actual_sources
    ]

    return all(
        any(
            expected.lower() in actual
            for actual in normalized_actual
        )
        for expected in expected_sources
    )

def calculate_keyword_coverpage(
        expected_keywords: list[str],
        answer: str,
) -> float:
    if not expected_keywords:
        return 1.0

    local_answer = answer.lower()

    matched_keywords = sum(
        keyword.lower() in local_answer
        for keyword in expected_keywords
    )

    return(
        matched_keywords/len(expected_keywords)
    )

def check_refusal(answer : str) -> bool:
    lower_answer = answer.lower()

    return any(
        marker in lower_answer
        for marker in REFUSAL_MARKERS
    )

def percentage(
    passed: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return round(
        passed / total * 100,
        1,
    )

def run_evaluation() -> None:
    cases = load_cases()
    results = []

    current_repository = None

    print(
        f"Running {len(cases)} "
        f"RepoLens evaluation cases...\n"
    )

    for index,case in enumerate(
        cases,
        start=1
    ):
        print(
            f"[{index}/{len(cases)}]"
            f"{case['name']}"
        )

        try:
            repository = case["repository"]

            if repository != current_repository:
                backend.prepare_project(repository)
                current_repository = repository

            thread_id = (
                f"evaluation-{uuid.uuid4()}"
            )

            config = {
                "configurable": {
                    "thread_id": thread_id
                },
                "recursion_limit": 30,
            }

            start_time = time.perf_counter()

            graph_result = backend.graph.invoke(
                {
                     "question": case["question"],
                    "question_type": "",
                    "classification_reason": "",
                    "answer": "",
                    "messages": [
                        HumanMessage(
                            content=case["question"]
                        )
                    ],
                },
                config=config,
            )

            latency = (
                time.perf_counter()
                - start_time
            )

            answer = graph_result.get(
                "answer",
                "",
            )

            actual_route = graph_result.get(
                 "question_type",
                "",
            )

            actual_sources = (
                backend.collect_sources(
                    graph_result["messages"]
                )
            )

            route_pass = (
                actual_route == case["expected_route"]
            )

            source_pass = check_sources(
                case["expected_sources"],
                actual_sources,
            )

            keyword_coverage = (
                calculate_keyword_coverpage(
                    case["expected_keywords"],
                    answer,
                )
            )

            keyword_pass = (
                keyword_coverage >= 0.6
            )

            if case["must_refuse"]:
                refusal_pass = check_refusal(
                    answer
                )
            else:
                refusal_pass = True

            case_pass = all(
                [
                     route_pass,
                    source_pass,
                    keyword_pass,
                    refusal_pass,
                ]
            )

            result = {
                "name": case["name"],
                "question": case["question"],
                "repository": repository,
                "passed": case_pass,
                "expected_route": (
                    case["expected_route"]
                ),
                "actual_route": actual_route,
                "route_pass": route_pass,
                "expected_sources": (
                    case["expected_sources"]
                ),
                "actual_sources": actual_sources,
                "source_pass": source_pass,
                "expected_keywords": (
                    case["expected_keywords"]
                ),
                "keyword_coverage": round(
                    keyword_coverage,
                    2,
                ),
                "keyword_pass": keyword_pass,
                "must_refuse": (
                    case["must_refuse"]
                ),
                "refusal_pass": refusal_pass,
                "latency_seconds": round(
                    latency,
                    2,
                ),
                "answer": answer,
                "error": None,
            }

            status = (
                "PASS"
                if case_pass
                else "FAIL"
            )

            print(
                f"{status} |"
                f"route = {actual_route} |"
                f"time = {latency:.2f}s\n"

            )

        
        except Exception as error:
            result = {
                "name": case["name"],
                "question": case["question"],
                "repository": case[
                    "repository"
                ],
                "passed": False,
                "error": str(error),
            }

            print(
                f"  ERROR: {error}\n"
            )

        results.append(result)

    total_cases = len(results)

    passed_cases = sum(
        result.get("passed", False)
        for result in results
    )

    route_results = [
        result
        for result in results
        if "route_pass" in result
    ]

    source_results = [
        result
        for result in results
        if result.get("expected_sources")
    ]

    keyword_results = [
        result
        for result in results
        if result.get("expected_keywords")
    ]

    refusal_results = [
        result
        for result in results
        if result.get("must_refuse")
    ]

    completed_results = [
        result
        for result in results
        if result.get("error") is None
    ]
    average_latency = 0.0

    if completed_results:
        average_latency = round(
            sum(
                result["latency_seconds"]
                for result in completed_results
            )
            /len(completed_results),
            2,
        )

    average_keyword_coverage = 0.0

    if keyword_results:
        average_keyword_coverage = round(
            sum(
                result["keyword_coverage"]
                for result in keyword_results
            )
            /len(keyword_results)
            *100,
            1,
        )

    summary = {
        "total_cases": total_cases,
        "passes_cases": passed_cases,
        "overall_pass_rate": percentage(
            passed_cases,
            total_cases,
        ),
        "routing_accuracy": percentage(
            sum(
                result["route_pass"]
                for result in route_results
            ),
            len(route_results),
        ),
        "source_accuracy": percentage(
            sum(
                result["source_pass"]
                for result in source_results
            ),
            len(source_results),
        ),
        "keyword_coverage": (
            average_keyword_coverage
        ),
        "refusal_accuracy": percentage(
            sum(
                result["refusal_pass"]
                for result in refusal_results
            ),
            len(refusal_results),
        ),
        "average_latency_seconds": (
            average_latency
        ),
    }

    output = {
        "summary":summary,
        "results": results

    }

    RESULTS_PATH.write_text(
        json.dumps(
            output,
            indent = 2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 55)
    print("RepoLens Evaluation Summary")
    print("=" * 55)

    print(
        f"Overall pass rate: "
        f"{summary['overall_pass_rate']}%"
    )

    print(
        f"Routing accuracy: "
        f"{summary['routing_accuracy']}%"
    )

    print(
        f"Source accuracy: "
        f"{summary['source_accuracy']}%"
    )

    print(
        f"Keyword coverage: "
        f"{summary['keyword_coverage']}%"
    )

    print(
        f"Refusal accuracy: "
        f"{summary['refusal_accuracy']}%"
    )

    print(
        f"Average latency: "
        f"{summary['average_latency_seconds']}s"
    )

    print(
        f"\nDetailed results saved to: "
        f"{RESULTS_PATH}"
    )

if __name__=="__main__":
    run_evaluation()
