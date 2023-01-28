import os
from aiogram.dispatcher import Dispatcher
from aiogram import Bot, types, executor
from aiogram.utils.helper import Helper, HelperMode, ListItem
import nest_asyncio
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import asyncio
from model import gen_output
from env import API_TOKEN

TOKEN = os.getenv('BOT_TOKEN')
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())




class TestStates(Helper):
    mode = HelperMode.snake_case

    TEST_STATE_0 = ListItem()
    TEST_STATE_1 = ListItem()
    TEST_STATE_2 = ListItem()
    TEST_STATE_3 = ListItem()
    TEST_STATE_4 = ListItem()
    
@dp.message_handler(state="*", commands=["start"])
async def start(m):
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    items = [types.KeyboardButton("Загрузить нестильную фотку"),
            ]
            
    for i in items:
        markup.add(i)
   
    await m.answer("\U0001f7e2 Привет. Это стильный бот для изменения стиля фотографии на целевую", reply_markup=markup)
    state = dp.current_state(user=m.from_user.id)
    await state.set_state(TestStates.all()[0])
    

    
@dp.message_handler(state=TestStates.TEST_STATE_0)
async def adddd(m):
    
    state = dp.current_state(user=m.from_user.id)
    
    await m.answer("Жду нестильное фото")
    await state.set_state(TestStates.all()[1])     
    
    
    
@dp.message_handler(content_types=['photo'], state=TestStates.TEST_STATE_1)
async def handle_docs_photo(message):

    await message.photo[-1].download(f'content_{message.from_user.id}.jpg')
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    items = [types.KeyboardButton("А теперь загрузить стиль"),
            ]
    for i in items:
        markup.add(i)
    
    await message.answer("Готово", reply_markup=markup)
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[2])     
    

    
    
@dp.message_handler(state=TestStates.TEST_STATE_2)
async def adddd(m):
    
    state = dp.current_state(user=m.from_user.id)
    
    await m.answer("Жду стильное фото")
    await state.set_state(TestStates.all()[3]) 
    
    

@dp.message_handler(content_types=['photo'], state=TestStates.TEST_STATE_3)
async def handle_docs_photo(message):

    await message.photo[-1].download(f'style_{message.from_user.id}.jpg')
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    items = [types.KeyboardButton("Получить результат"),
            ]
    for i in items:
        markup.add(i)
        
    await message.answer("Ждем преображения. 3 минуты, сэр", reply_markup=markup)
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[4])     
    
    
@dp.message_handler(state=TestStates.TEST_STATE_4)
async def handle_docs_photo(message):
    
    gen_output(message.from_user.id)
    await asyncio.sleep(60*2)

    await message.answer_photo(types.InputFile(f'result_{message.from_user.id}.png'))
    
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[1])           


nest_asyncio.apply()
executor.start_polling(dp)