# Test pydantic validation (2 tests thus far)
from app.schemas import SearchRequest, ChatRequest


def test_SearchRequest():
    query = "Yo!"
    mode = "semantic"
    filter = {"year": 2025}
    k = 3
    response = SearchRequest(
        query= query, 
        mode= mode, 
        filters= filter,
        top_k= k
    )

    assert response.query == "Yo!"
    assert response.mode == "semantic"
    assert response.top_k == 3

def test_ChatRequest():
    thread_id = "test"
    msg = "Well well well"
    mode = "lexical"
    filter = {"year": 2025}
    response = ChatRequest(
        thread_id= thread_id, 
        message= msg,
        mode= mode,
        filters= filter    
        )

    assert response.thread_id == "test"
    assert response.message == "Well well well"
    assert response.filters == {"year": 2025}