from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gpt_researcher import GPTResearcher  # Assuming this is the library you are using
import asyncio

# Define the FastAPI app
app = FastAPI()

# Define the input model for the query
class QueryRequest(BaseModel):
    query: str

# Endpoint to handle research queries
@app.post("/generate_report/")
async def generate_report(request: QueryRequest):
    # Create an instance of GPTResearcher with the given query
    researcher = GPTResearcher(query=request.query, report_type="very_short_report")

    # Conduct research asynchronously
    research_result = await researcher.conduct_research()

    # Write the research report asynchronously
    report = await researcher.write_report()

    # Return the report as a response
    return {"query": request.query, "report": report}


# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

