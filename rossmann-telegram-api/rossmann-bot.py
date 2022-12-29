### IMPORT LIBRARIES

import pandas as pd
import os
import json
import requests
## Biblioteca para construir interfaces(app) Web em Python para construir API´s
from flask import Flask, request, Response


# constants BOT Telegram
TOKEN = '5913411341:AAGPaV0Hb9lSmnpheI1cjVN-CpPAY9_LURQ'

# Info about the Bot
#https://api.telegram.org/bot5913411341:AAGPaV0Hb9lSmnpheI1cjVN-CpPAY9_LURQ/getMe

# get updates
#https://api.telegram.org/bot5913411341:AAGPaV0Hb9lSmnpheI1cjVN-CpPAY9_LURQ/getUpdates

# send message (método + ? passar parâmetros -> chat_id e text )
#https://api.telegram.org/bot5913411341:AAGPaV0Hb9lSmnpheI1cjVN-CpPAY9_LURQ/sendMessage?chat_id=1069746879&text=Hi Alan, I am doing good, tks!

# Webhook (conexão da internet com local para receber dado do numero da loja)
#https://api.telegram.org/bot5913411341:AAGPaV0Hb9lSmnpheI1cjVN-CpPAY9_LURQ/setWebhook?url=https://3d584ca0b1335e.lhr.life

# Webhook Render (definir que a mensagem escrita no BOT Telegram(/22 numero da loja) vai direto pra API como POST e retorna a predição da loja desejada)
#https://api.telegram.org/bot5913411341:AAGPaV0Hb9lSmnpheI1cjVN-CpPAY9_LURQ/setWebhook?url=https://bot-telegram-input.onrender.com



### FUNCTIONS

def send_message( chat_id, text ):
    url = 'https://api.telegram.org/bot{}/'.format( TOKEN )
    url = url + 'sendMessage?chat_id={}'.format( chat_id )

    # Escrevendo mensagem na conversa do Bot no telegram
    r = requests.post( url, json = {'text': text } )
    print( 'Status Code {}'.format( r.status_code ) )

    return None


def load_dataset( store_id ):

    #loading test dataset
    df_test_raw = pd.read_csv('data/test.csv')
    df_store_raw = pd.read_csv('data/store.csv')

    ## merge test dataset + store
    df_test = df_test_raw.merge(df_store_raw, how = "left", on = "Store")

    ## choose store for prediction
    df_test = df_test[ df_test['Store'] == store_id ]

    # Teste para verificar se loja indicada existe no dataset de teste
    if not df_test.empty:

        # Manipulation

        ## remove closed days
        df_test = df_test[ df_test['Open'] != 0 ]
        ## select just samples without NA
        df_test = df_test[ df_test['Open'].notnull() ]
        ## exclude column "Id"
        df_test = df_test.drop('Id', axis = 1)

        # convert Dataframe to json for send comunication between the systems
        data = json.dumps( df_test.to_dict( orient = 'records' ) )

    else:
        data = 'error'

    return data


def predict( data ):

    # API Call

    # url em servidor na nuvem
    url = 'https://rossmann-predict-sales-project.onrender.com/rossmann/predict'
    # Formato dos dados que serão enviados
    header = {'Content-type': 'application/json'}
    # dado a ser enviado
    data = data

    r = requests.post(url = url, data = data, headers = header)
    print( f'Status Code { r.status_code }' )

    ## Predição diária de vendas durante 6 semanas para as lojas selecionadas
    d1 = pd.DataFrame( r.json(), columns = r.json()[0].keys() )

    return d1

def parse_message( message ):

    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    # preparação para receber n loja da forma correta
    store_id = store_id.replace( '/' , '' )

    try: # forçar o texto a virar numero inteiro que é o certo a ser recebido
        store_id = int( store_id )

    except ValueError:
        store_id = 'error'

    return chat_id, store_id


### API INITIALIZE

app = Flask( __name__ )

# decorador para gerar endpoint na raíz('/') com método POST (envia algum dado para poder receber)
## método GET (pede algum dado para poder receber)
@app.route('/', methods = ['GET' ,'POST'])
## Função index irá rodar toda vez que o endpoint '/' for acionado passando um dado
def index():
    if request.method == 'POST':
        # mensagem digitada pelo usuário no BOT do Telegram em JSON
        message = request.get_json()
        ## Função para análise e extração da mensagem digitada pelo usuário no BOT do Telegram
        chat_id, store_id = parse_message( message )

        if store_id != 'error':
            # loading data
            data = load_dataset( store_id )

            if data != 'error':

                # prediction
                d1 = predict( data )

                # calculation

                ## Soma da Predição diária de vendas para as 6 semanas da loja selecionada
                d2 = d1[['store','prediction']].groupby('store').sum().reset_index()

                # send message
                msg = 'A loja nº {} irá vender US$ {:,.2f} dólares nas próximas 6 semanas.'.format( d2['store'].values[0] ,
                                                                                                    d2['prediction'].values[0] )

                send_message( chat_id, msg )
                # Quando o processo chega ao fim deve-se enviar essa mensagem pra dizer pra API que está tudo OK e não ficar requisitando e extraindo dado toda hora
                return Response('Ok', status = 200)

            else:
                send_message( chat_id, 'ID Loja não disponível, digite outro ID de loja.' )
                # Quando o processo chega ao fim deve-se enviar essa mensagem pra dizer pra API que está tudo OK e não ficar requisitando e extraindo dado toda hora
                return Response('Ok', status = 200)


        else:

            send_message( chat_id, 'ID Loja está errado! - confirme que você está digitando corretamento o número da loja.' )
            # Quando o processo chega ao fim deve-se enviar essa mensagem pra dizer pra API que está tudo OK e não ficar requisitando e extraindo dado toda hora
            return Response('Ok', status = 200)


    else: # Mensagem caso o usuário não passar nenhum dado e acessar o endpoint
        return '<h1> Rossmann Telegram BOT </h1>'




if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( host = '0.0.0.0', port = port )

# 192.168.0.6 -> endereço IPv4 pc local
