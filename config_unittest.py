tests = {'OPENAI':
    {
        'Model-ok'              : ['test_model','gpt-4o',200],
        'Model-bad'             : ['test_model','failing-llm',500],
        'Reload'                : ['test_reload','',200],
        'Clear'                 : ['test_clear','',200],
        'Cache'                 : ['test_cache',{
            'answer1' :"User:content='who wrote rag service?'",
            'answer2'           :"AI:content='RAG Service was developed by L1Blom.'"
        }, 200],
        'Temperature-ok'        : ['test_temperature',0.0,200],
        'Temperature-too-low'   : ['test_temperature',-0.6,500],
        'Temperature-too-high'  : ['test_temperature',2.1,500],
        'Prompt-text'           : ['test_prompt_text',{
            'prompt' : "prompt=who wrote rag service?",
            'answer' : "RAG Service was developed by Leen Blom."
        }, 200],
        'Prompt-PDF'            : ['test_prompt_pdf',{
            'prompt' : "prompt=how many watchers has this GitHub library?",
            'answer' : "The GitHub repository for the RAG Service has 2 watchers."
        }, 200],
        'Image'                 : ['test_image',{
            'prompt' : "what is written",
            'answer' : "OpenAI."
        },200]
    },
    'GROQ':
    {
        'Model-ok'              : ['test_model','llama-3.1-8b-instant',200],
        'Model-bad'             : ['test_model','failing-llm',500],
        'Reload'                : ['test_reload','',200],
        'Clear'                 : ['test_clear','',200],
        'Cache'                 : ['test_cache',{
            'answer1' :"User:content='who wrote rag service?'",
            'answer2'           :"AI:content='RAG Service was developed by Leen Blom.'"
        }, 200],
        'Temperature-ok'        : ['test_temperature',0.0,200],
        'Temperature-too-low'   : ['test_temperature',-0.6,500],
        'Temperature-too-high'  : ['test_temperature',2.1,500],
        'Prompt-text'           : ['test_prompt_text',{
            'prompt' : "prompt=who wrote rag service?",
            'answer' : "RAG Service was written by Leen Blom."
        }, 200],
        'Prompt-PDF'            : ['test_prompt_pdf',{
            'prompt' : "prompt=how many watchers has this GitHub library?",
            'answer' : "The GitHub repository for the RAG Service has 2 watchers."
        }, 200],
        'Image'                 : ['test_image',{
            'prompt' : "N/A",
            'answer' : "N/A"
        },500]
    }
}
