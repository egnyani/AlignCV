FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    fonts-crosextra-carlito \
    fonts-crosextra-caladea \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f -v

# Configure LibreOffice font substitution: Calibri → Carlito (metric-compatible)
RUN mkdir -p /root/.config/libreoffice/4/user && \
    cat > /root/.config/libreoffice/4/user/registrymodifications.xcu << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry"
           xmlns:xs="http://www.w3.org/2001/XMLSchema"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <item oor:path="/org.openoffice.VCL/Settings/FontSubstitutions">
    <node oor:name="1" oor:op="replace">
      <prop oor:name="ReplaceFont" oor:type="xs:string"><value>Calibri</value></prop>
      <prop oor:name="SubstituteFont" oor:type="xs:string"><value>Carlito</value></prop>
    </node>
  </item>
</oor:items>
EOF

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8000
EXPOSE 8000

CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
