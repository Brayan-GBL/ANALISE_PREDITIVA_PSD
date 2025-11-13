import requests
from lxml import etree
# === CONFIGURAÇÃO ===
CNPJ = "20736476000100"  # CNPJ da empresa a consultar
UF = "MG"  # Sigla do estado
CERT_PATH = "certificado.pfx"  # Caminho para seu certificado A1 (.pfx)
CERT_PASSWORD = "SENHA_DO_CERTIFICADO"
# Endpoint do serviço oficial (SEFAZ-MG neste exemplo)
URL = "https://nfe.fazenda.mg.gov.br/nfe2/services/CadConsultaCadastro4"
# === XML da requisição SOAP ===
xml_envelope = f"""<?xml version="1.0" encoding="utf-8"?>
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
                  xmlns:cad="http://www.portalfiscal.inf.br/nfe">
   <soapenv:Header/>
   <soapenv:Body>
      <cad:consCad versao="4.00" xmlns="http://www.portalfiscal.inf.br/nfe">
         <infCons>
            <xServ>CONS-CAD</xServ>
            <UF>{UF}</UF>
            <CNPJ>{CNPJ}</CNPJ>
         </infCons>
      </cad:consCad>
   </soapenv:Body>
</soapenv:Envelope>
"""
# === ENVIO DA REQUISIÇÃO ===
try:
    from requests_pkcs12 import post
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8"
    }
    response = post(
        URL,
        data=xml_envelope.encode("utf-8"),
        headers=headers,
        pkcs12_filename=CERT_PATH,
        pkcs12_password=CERT_PASSWORD,
        timeout=30
    )
    # === Verificação de resposta ===
    if response.status_code == 200:
        xml = etree.fromstring(response.content)
        ns = {"nfe": "http://www.portalfiscal.inf.br/nfe"}
        # Extraindo campos principais
        cStat = xml.xpath("//nfe:cStat/text()", namespaces=ns)
        xMotivo = xml.xpath("//nfe:xMotivo/text()", namespaces=ns)
        xNome = xml.xpath("//nfe:xNome/text()", namespaces=ns)
        IE = xml.xpath("//nfe:IE/text()", namespaces=ns)
        UF_resp = xml.xpath("//nfe:UF/text()", namespaces=ns)
        cSit = xml.xpath("//nfe:cSit/text()", namespaces=ns)
        print("=== Resultado da Consulta ===")
        print(f"Status: {cStat[0] if cStat else ''} - {xMotivo[0] if xMotivo else ''}")
        print(f"Razão Social: {xNome[0] if xNome else ''}")
        print(f"IE: {IE[0] if IE else ''}")
        print(f"UF: {UF_resp[0] if UF_resp else ''}")
        print(f"Situação Cadastral: {cSit[0] if cSit else ''}")
    else:
        print(f"Erro HTTP {response.status_code}: {response.text}")
except Exception as e:
    print(f"Erro na consulta: {e}")
