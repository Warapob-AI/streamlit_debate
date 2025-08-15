from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import  RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.agents import AssistantAgent
from googlesearch import search
import json
from dotenv import load_dotenv
load_dotenv()
import asyncio 
import os 

def GoogleSearch(Query: str) -> str: 
    results_list = []
    
    try:
        SearchResults = search(Query, num_results=10, advanced=True, sleep_interval=1)
        
        for result in SearchResults: 
            # กรองลิงก์ที่ไม่ต้องการออก
            if "facebook" in result.url or "wikipedia" in result.url:
                continue
            
            # 3. สร้าง Dictionary สำหรับแต่ละผลลัพธ์
            result_data = {
                "title": result.title,
                "url": result.url,
                "description": result.description
            }
            
            # 4. เพิ่ม Dictionary ที่สร้างลงในลิสต์
            results_list.append(result_data)
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาดระหว่างการค้นหา: {e}")
        # อาจจะ return error object กลับไปก็ได้
        return {"error": str(e)}


    json_output = json.dumps(results_list, ensure_ascii=False, indent=4)
    
    return json_output

async def teamConfig(task_user):

    model_gemini = OpenAIChatCompletionClient(
        model = 'gemini-2.5-flash',
        api_key = os.getenv('GEMINI_API_KEY'),
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            'family': 'gemini-2.5-pro',
            'structured_output': False
        }
    )
    model_ollama = OllamaChatCompletionClient(
        model="gpt-oss:latest",
        host=os.getenv('HOST_OLLAMA'),
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            'family': 'gemini-2.5-pro',
            'structured_output': False
        }
    )

    PlanningAgent = AssistantAgent(
        name = 'PlanningAgent',
        model_client = model_gemini,
        system_message = f"""
        
        คุณคือ AI AGENT ที่มีหน้าที่เป็นคนควบคุมการ Debate ครั้งนี้ โดยคุณจะต้องเป็นกรรมการ และพยายามตัดสินว่าใครเป็นฝ่ายถูกหรือผิด หรือมีแนวโน้มที่จะเป็นความจริงมากกว่ากัน
        โดยที่คุณจะต้องมอบหมายให้กับ Jack ก่อนเป็นผู้เอนเอียงไปทางบวก และ John เป็นผู้เอนเอียงไปทางลบ 
        
        โดยที่เราจะจัด Debate ต่อเนื่องกัน 3 ครั้ง และจบที่หลังจากที่ John และ Jack Debate มาแล้วครบ 3 ครั้ง เราถึงสรุปรอบท้าย
        โดยเริ่มจากคุณเป็นกรรมการ และเริ่มจาก Jack จากนั้นไปที่ John และไปที่คุณ วนไปเรื่อย ๆ จนกว่าจะครบ 3 ครั้ง
        
        นายไม่ใช่คนทำงานทั้งหมด แค่เป็นกรรมการ แล้วจบหน้าที่ของนาย แล้วโยนไปให้ Jack แล้วค่อยรับต่อจาก John
        วิเคราะห์จาก task :  {task_user}
        
        ตอบเป็นภาษาไทยเท่านั้น
        
        เมื่อเกริ่นเสร็จแล้ว ให้มอบหมายหน้าที่ไปที่ Jack ทันที

        """
    )

    JackAgent = AssistantAgent(
        name = 'JackAgent',
        model_client = model_gemini, 
        tools = [GoogleSearch], 
        system_message = f"""
        
        คุณคือ AI AGENT ชื่อ Jack ที่มีหน้าที่ Debate โดยเอนเอียงตามหลักเหตุและผลไปทางบวก เพื่อสู้กับ John ที่เป็นคน Debate ของคุณ ซึ่ง John จะเอนเอียงไปทางลบ 
        คุณต้อง Debate โดยคิดว่ามันจะมีแนวโน้มที่จะเป็นความจริงมากเท่าไหร่ 

        เมื่อคุณ Debate แต่ละรอบเสร็จ คุณต้องส่งไปให้ John เป็นผู้ Debate ต่อไป และเมื่อรอบถัด ๆ ไป คุณต้องจำคำพูดของเขามาถกเถียงกันอีกที 
        
        โดยที่เราจะจัด Debate ต่อเนื่องกัน 3 ครั้ง และจบที่ครั้งที่ 3 
        
        วิเคราะห์จาก task :  {task_user}
        
        ตอบเป็นภาษาไทยเท่านั้น
        """,
        reflect_on_tool_use=True
    )

    JohnAgent = AssistantAgent(
        name = 'JohnAgent',
        model_client = model_gemini, 
        tools = [GoogleSearch], 
        system_message = f"""
        
        คุณคือ AI AGENT ชื่อ John ที่มีหน้าที่ Debate โดยเอนเอียงตามหลักเหตุและผลไปทางลบ เพื่อสู้กับ Jack ที่เป็นคน Debate ของคุณ ซึ่ง Jack จะเอนเอียงไปทางบวก 
        คุณต้อง Debate โดยคิดว่ามันจะมีแนวโน้มที่จะเป็นความจริงมากเท่าไหร่ 

        เมื่อคุณ Debate แต่ละรอบเสร็จ คุณต้องส่งไปให้ PlanningAgent เป็นกรรมการและตัดสินก่อนเข้าสู่รอบถัดไป และเมื่อรอบถัด ๆ ไป คุณต้องจำคำพูดของเขามาถกเถียงกันอีกที 
        
        โดยที่เราจะจัด Debate ต่อเนื่องกัน 3 ครั้ง และจบที่ครั้งที่ 3 
        
        วิเคราะห์จาก task :  {task_user}
        
        ตอบเป็นภาษาไทยเท่านั้น
        """,
        reflect_on_tool_use=True,
    )

    team = SelectorGroupChat(
        participants=[PlanningAgent, JackAgent, JohnAgent],
        termination_condition=TextMentionTermination('TERMINATE'),
        model_client = model_gemini,
        max_turns=10
    )
    
    return team 

from autogen_agentchat.base import TaskResult 

async def debate(team): 
    async for message in team.run_stream(task="Start the debate!"): 
        if (isinstance(message, TaskResult)):
            message = (f'Stopping reason: {message.stop_reason}')
            yield message 

        else: 
            message = (f'{message.source}: {message.content}')
            yield message 


from autogen_agentchat.messages import TextMessage

import streamlit as st 

# ===================================================================================
# FRONT-END CODE 
# ===================================================================================
st.header("Agents Debate!")

topic = st.text_input("Enter the topic of the debate")

clicked = st.button("Start", type="primary")

chat = st.container()

if (clicked): 
    chat.empty() 

    async def main(): 
        team = await teamConfig(topic)
        with chat: 
            with st.spinner("Generate Loading..."): 
                async for message in debate(team): 
                    if (message.startswith("PlanningAgent")): 
                        with st.chat_message('user'):
                            st.markdown(message)

                    elif (message.startswith("JackAgent")): 
                        with st.chat_message('ai'):
                            st.markdown(message)

                    elif (message.startswith("JohnAgent")): 
                        with st.chat_message('ai'):
                            st.markdown(message)

    asyncio.run(main())