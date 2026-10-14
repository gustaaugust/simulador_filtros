import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, rfftfreq

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variáveis
sinal_original = None
sinal_filtrado = None
sinal_desejado = None
componente1 = None
componente2 = None
f1_atual = None
f2_atual = None
tempo = None
frequencia = None
fs_atual = None

# =========================== PROCESSAMENTO E VALORES ===========================
def gerar_sinal():
    global sinal_original, sinal_desejado, componente1, componente2
    global f1_atual, f2_atual, tempo, fs_atual

    try:
        # Obtém os valores da interface
        fs = float(entry_freq_samp.get())
        ruido_nivel = float(entry_ruido.get())
        f1 = float(entry_freq_sinal.get())
        f2 = float(entry_freq2.get())

        amplitude1 = 1.0
        amplitude2 = 0.5
        duracao = 5.0

        # Verificações gerais
        if fs <= 0:
            raise ValueError("A frequência de amostragem deve ser maior que zero.")

        if duracao <= 0:
            raise ValueError("A duração deve ser maior que zero.")

        if ruido_nivel < 0:
            raise ValueError("O nível de ruído não pode ser negativo.")

        if f1 <= 0:
            raise ValueError("A frequência do sinal deve ser maior que zero.")

        # A frequência 2 pode ser zero. Nesse caso, ela é desativada.
        if f2 < 0:
            raise ValueError("A frequência 2 não pode ser negativa.")

        # Frequência de Nyquist
        if f1 >= fs / 2:
            raise ValueError(
                "A frequência do sinal deve ser menor que a frequência de Nyquist."
            )

        if f2 > 0 and f2 >= fs / 2:
            raise ValueError(
                "A frequência 2 deve ser menor que a frequência de Nyquist."
            )

        # Criação do vetor de tempo
        tempo = np.arange(0, duracao, 1 / fs)

        # Primeira componente do sinal
        componente1 = amplitude1 * np.sin(2 * np.pi * f1 * tempo)

        # Segunda componente do sinal. Se f2 = 0, ela fica zerada.
        if f2 == 0:
            componente2 = np.zeros_like(tempo)
        else:
            componente2 = amplitude2 * np.sin(2 * np.pi * f2 * tempo)

        # Guarda as frequências realmente usadas para gerar este sinal
        f1_atual = f1
        f2_atual = f2

        # O sinal desejado será definido ao aplicar o filtro, porque depende
        # do tipo de filtro e das frequências que devem permanecer.
        sinal_desejado = None

        # Sinal do sensor antes da adição do ruído branco
        sinal_sensor = componente1 + componente2

        # Geração do ruído branco
        ruido = np.random.normal(0, ruido_nivel, len(tempo))

        # Sinal final recebido pelo sistema
        sinal_original = sinal_sensor + ruido

        # Guarda a frequência de amostragem usada para gerar este sinal
        fs_atual = fs

        messagebox.showinfo("Sucesso", "Sinal gerado com sucesso!")

    except ValueError as e:
        messagebox.showerror("Erro", f"Valor inválido: {e}")

# Função para filtrar o sinal
def aplicar_filtro():
    global sinal_filtrado, sinal_desejado, frequencia, fs_atual

    if sinal_original is None:
        messagebox.showwarning(
            "Aviso",
            "Primeiro gere o sinal."
        )
        return

    try:
        fs = float(entry_freq_samp.get())

        # Impede aplicar o filtro com uma frequência de amostragem diferente
        # daquela usada para gerar o sinal.
        if fs_atual is None:
            raise ValueError("Gere o sinal novamente antes de aplicar o filtro.")

        if not np.isclose(fs, fs_atual):
            raise ValueError(
                "A frequência de amostragem foi alterada. Gere o sinal novamente."
            )

        # Usa exatamente a frequência de amostragem do sinal gerado
        fs = fs_atual

        ordem = int(entry_ordem.get())
        tipo_filtro = cb_filtro.get()

        # Verificações gerais
        if fs <= 0:
            raise ValueError("A frequência de amostragem deve ser maior que zero.")

        if ordem <= 0:
            raise ValueError("A ordem do filtro deve ser maior que zero.")

        nyquist = fs / 2

        # Escolha do tipo de filtro
        if tipo_filtro == "Passa-Baixa":
            fc = float(entry_freq_corte.get())

            if fc <= 0:
                raise ValueError("A frequência de corte deve ser maior que zero.")

            if fc >= nyquist:
                raise ValueError(
                    "A frequência de corte deve ser menor que a frequência de Nyquist."
                )

            b, a = butter(
                ordem,
                fc,
                btype="low",
                fs=fs
            )

        elif tipo_filtro == "Passa-Alta":
            fc = float(entry_freq_corte.get())

            if fc <= 0:
                raise ValueError("A frequência de corte deve ser maior que zero.")

            if fc >= nyquist:
                raise ValueError(
                    "A frequência de corte deve ser menor que a frequência de Nyquist."
                )

            b, a = butter(
                ordem,
                fc,
                btype="high",
                fs=fs
            )

        elif tipo_filtro == "Passa-Faixa":
            freq_min = float(entry_freq_min.get())
            freq_max = float(entry_freq_max.get())

            if freq_min <= 0 or freq_max <= 0:
                raise ValueError(
                    "As frequências mínima e máxima devem ser maiores que zero."
                )

            if freq_min >= freq_max:
                raise ValueError(
                    "A frequência mínima deve ser menor que a frequência máxima."
                )

            if freq_max >= nyquist:
                raise ValueError(
                    "A frequência máxima deve ser menor que a frequência de Nyquist."
                )

            b, a = butter(
                ordem,
                [freq_min, freq_max],
                btype="bandpass",
                fs=fs
            )

        elif tipo_filtro == "Rejeita-Faixa":
            freq_min = float(entry_freq_min.get())
            freq_max = float(entry_freq_max.get())

            if freq_min <= 0 or freq_max <= 0:
                raise ValueError(
                    "As frequências mínima e máxima devem ser maiores que zero."
                )

            if freq_min >= freq_max:
                raise ValueError(
                    "A frequência mínima deve ser menor que a frequência máxima."
                )

            if freq_max >= nyquist:
                raise ValueError(
                    "A frequência máxima deve ser menor que a frequência de Nyquist."
                )

            b, a = butter(
                ordem,
                [freq_min, freq_max],
                btype="bandstop",
                fs=fs
            )

        else:
            raise ValueError(
                "Este tipo de filtro ainda não está disponível."
            )

        # Define o sinal desejado IDEAL: somente as componentes que o filtro
        # escolhido deveria preservar. Isso faz a linha de referência mudar
        # corretamente para Passa-Baixa, Passa-Alta, Passa-Faixa e Rejeita-Faixa.
        sinal_desejado = np.zeros_like(tempo)

        if tipo_filtro == "Passa-Baixa":
            if f1_atual < fc:
                sinal_desejado += componente1
            if f2_atual > 0 and f2_atual < fc:
                sinal_desejado += componente2

        elif tipo_filtro == "Passa-Alta":
            if f1_atual > fc:
                sinal_desejado += componente1
            if f2_atual > 0 and f2_atual > fc:
                sinal_desejado += componente2

        elif tipo_filtro == "Passa-Faixa":
            if freq_min < f1_atual < freq_max:
                sinal_desejado += componente1
            if f2_atual > 0 and freq_min < f2_atual < freq_max:
                sinal_desejado += componente2

        elif tipo_filtro == "Rejeita-Faixa":
            if f1_atual < freq_min or f1_atual > freq_max:
                sinal_desejado += componente1
            if f2_atual > 0 and (f2_atual < freq_min or f2_atual > freq_max):
                sinal_desejado += componente2

        # Aplicação do filtro
        sinal_filtrado = filtfilt(
            b,
            a,
            sinal_original
        )

        # Cálculo da FFT
        n = len(sinal_original)

        fft_original = (2 / n) * np.abs(
            rfft(sinal_original)
        )

        fft_filtrado = (2 / n) * np.abs(
            rfft(sinal_filtrado)
        )

        # Frequências da FFT
        frequencia = rfftfreq(
            n,
            1 / fs
        )

        # Atualiza os gráficos
        atualizar_graficos(
            fft_original,
            fft_filtrado
        )

    except ValueError as e:
        messagebox.showerror(
            "Erro",
            f"Valor inválido: {e}"
        )

def limpar_dados():
    global sinal_original, sinal_filtrado, sinal_desejado
    global componente1, componente2, f1_atual, f2_atual, tempo, frequencia, fs_atual

    sinal_original = None
    sinal_filtrado = None
    sinal_desejado = None
    componente1 = None
    componente2 = None
    f1_atual = None
    f2_atual = None
    tempo = None
    frequencia = None
    fs_atual = None

    # Limpa ou reseta os campos da tela
    entry_freq_sinal.delete(0, tk.END)
    entry_freq_sinal.insert(0, "5.0")

    entry_freq2.delete(0, tk.END)
    entry_freq2.insert(0, "15.0")

    entry_freq_corte.delete(0, tk.END)
    entry_freq_corte.insert(0, "10.0")

    entry_freq_min.delete(0, tk.END)
    entry_freq_min.insert(0, "4.0")

    entry_freq_max.delete(0, tk.END)
    entry_freq_max.insert(0, "10.0")

    entry_freq_samp.delete(0, tk.END)
    entry_freq_samp.insert(0, "100.0")

    entry_ordem.delete(0, tk.END)
    entry_ordem.insert(0, "2")

    entry_ruido.delete(0, tk.END)
    entry_ruido.insert(0, "0.1")

    if eixos is not None:
        limpar_graficos()

    messagebox.showinfo("Limpeza", "Dados reiniciados com sucesso!")

def atualizar_campos_interface(event=None):
    tipo_filtro = cb_filtro.get()

    # A frequência 2 fica disponível para todos os filtros.
    # O usuário pode colocar 0 para desativá-la.
    label_freq2.grid(row=2, column=0, sticky="w", pady=8)
    entry_freq2.grid(row=2, column=1, sticky="ew", pady=8)

    if tipo_filtro in ["Passa-Faixa", "Rejeita-Faixa"]:
        # Esconde a frequência de corte única
        lbl_freq_corte.grid_remove()
        entry_freq_corte.grid_remove()

        # Mostra os dois limites de corte
        lbl_freq_min.grid(row=3, column=0, sticky="w", pady=8)
        entry_freq_min.grid(row=3, column=1, sticky="ew", pady=8)

        lbl_freq_max.grid(row=4, column=0, sticky="w", pady=8)
        entry_freq_max.grid(row=4, column=1, sticky="ew", pady=8)

    else:
        # Passa-Baixa e Passa-Alta usam uma frequência de corte
        lbl_freq_corte.grid(row=3, column=0, sticky="w", pady=8)
        entry_freq_corte.grid(row=3, column=1, sticky="ew", pady=8)

        # Esconde os limites usados pelos filtros de faixa
        lbl_freq_min.grid_remove()
        entry_freq_min.grid_remove()

        lbl_freq_max.grid_remove()
        entry_freq_max.grid_remove()

# ======================= CRIAÇÃO DOS GRÁFICOS ============================
def atualizar_graficos(fft_original, fft_filtrado):
    global figura, eixos, canvas

    for eixo in eixos:
        eixo.clear()

    # GRÁFICO 1 - SINAL ANTES DA FILTRAGEM
    eixos[0].plot(
        tempo,
        sinal_original,
        label="Sinal original"
    )

    # Linha de referência: sinal ideal que o filtro deve preservar
    eixos[0].plot(
        tempo,
        sinal_desejado,
        label="Sinal desejado",
        color="orange",
        linewidth=2
    )

    eixos[0].set_title(
        "Sinal Antes da Filtragem"
    )

    eixos[0].set_xlabel(
        "Tempo (s)"
    )

    eixos[0].set_ylabel(
        "Amplitude"
    )

    eixos[0].legend()
    eixos[0].grid(True)

    # GRÁFICO 2 - SINAL DEPOIS DA FILTRAGEM
    eixos[1].plot(
        tempo,
        sinal_filtrado,
        label="Sinal filtrado"
    )

    # O mesmo sinal ideal é mostrado para comparar com o resultado filtrado
    eixos[1].plot(
        tempo,
        sinal_desejado,
        label="Sinal desejado",
        color="orange",
        linewidth=2
    )

    eixos[1].set_title(
        "Sinal Depois da Filtragem"
    )

    eixos[1].set_xlabel(
        "Tempo (s)"
    )

    eixos[1].set_ylabel(
        "Amplitude"
    )

    eixos[1].legend()
    eixos[1].grid(True)

    # GRÁFICO 3 - FFT DO SINAL ORIGINAL
    eixos[2].plot(
        frequencia,
        fft_original,
        label="FFT original"
    )

    eixos[2].set_title(
        "FFT do Sinal Original"
    )

    eixos[2].set_xlabel(
        "Frequência (Hz)"
    )

    eixos[2].set_ylabel(
        "Amplitude"
    )

    eixos[2].set_xlim(
        0,
        min(50, max(frequencia))
    )

    eixos[2].legend()
    eixos[2].grid(True)

    # GRÁFICO 4 - FFT DO SINAL FILTRADO
    eixos[3].plot(
        frequencia,
        fft_filtrado,
        label="FFT filtrada"
    )

    eixos[3].set_title(
        "FFT do Sinal Filtrado"
    )

    eixos[3].set_xlabel(
        "Frequência (Hz)"
    )

    eixos[3].set_ylabel(
        "Amplitude"
    )

    eixos[3].set_xlim(
        0,
        min(50, max(frequencia))
    )

    eixos[3].legend()
    eixos[3].grid(True)

    # Ajusta os gráficos
    figura.tight_layout()

    # Atualiza a interface
    canvas.draw()

# FUNÇÃO PARA LIMPAR
def limpar_graficos():
    for eixo in eixos:
        eixo.clear()
        eixo.set_xlabel("")
        eixo.set_ylabel("")
        eixo.set_title("")

    figura.tight_layout()
    canvas.draw()

# =========================== CONFIG. JANELA ===========================
root = tk.Tk()
root.title("Simulador de Filtros")
root.geometry("1100x650")
root.resizable(False, False)

# Divisão dos controles dos gráficos
main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# --- Coluna dos controles ---
frame_esq = ttk.Frame(main_pane)
main_pane.add(frame_esq, weight=1)

# --- Controles ---
frame_controles = ttk.LabelFrame(
    frame_esq,
    text=" Parâmetros do Filtro ",
    padding=15
)
frame_controles.pack(fill="both", expand=True, padx=15, pady=15)
frame_controles.columnconfigure(1, weight=1)

# Tipo de filtro
lbl_tipo = ttk.Label(frame_controles, text="Tipo de Filtro:")
lbl_tipo.grid(row=0, column=0, sticky="w", pady=8)

cb_filtro = ttk.Combobox(
    frame_controles,
    values=["Passa-Baixa", "Passa-Alta", "Passa-Faixa", "Rejeita-Faixa"],
    state="readonly"
)
cb_filtro.grid(row=0, column=1, sticky="ew", pady=8)
cb_filtro.current(0)
cb_filtro.bind("<<ComboboxSelected>>", atualizar_campos_interface)

# Frequência do sinal 1
lbl_freq_sinal = ttk.Label(
    frame_controles,
    text="Frequência do Sinal (Hz):"
)
lbl_freq_sinal.grid(row=1, column=0, sticky="w", pady=8)

entry_freq_sinal = ttk.Entry(frame_controles)
entry_freq_sinal.grid(row=1, column=1, sticky="ew", pady=8)
entry_freq_sinal.insert(0, "5.0")

# Frequência do sinal 2
label_freq2 = ttk.Label(
    frame_controles,
    text="Frequência 2 (Hz) - 0 desativa:"
)
label_freq2.grid(row=2, column=0, sticky="w", pady=8)

entry_freq2 = ttk.Entry(frame_controles)
entry_freq2.grid(row=2, column=1, sticky="ew", pady=8)
entry_freq2.insert(0, "15.0")

# Frequência de corte única - Passa-Baixa e Passa-Alta
lbl_freq_corte = ttk.Label(
    frame_controles,
    text="Frequência de Corte (Hz):"
)
lbl_freq_corte.grid(row=3, column=0, sticky="w", pady=8)

entry_freq_corte = ttk.Entry(frame_controles)
entry_freq_corte.grid(row=3, column=1, sticky="ew", pady=8)
entry_freq_corte.insert(0, "10.0")

# Frequência mínima - Passa-Faixa e Rejeita-Faixa
lbl_freq_min = ttk.Label(
    frame_controles,
    text="Frequência Mínima de Corte (Hz):"
)
lbl_freq_min.grid(row=3, column=0, sticky="w", pady=8)

entry_freq_min = ttk.Entry(frame_controles)
entry_freq_min.grid(row=3, column=1, sticky="ew", pady=8)
entry_freq_min.insert(0, "4.0")

# Frequência máxima - Passa-Faixa e Rejeita-Faixa
lbl_freq_max = ttk.Label(
    frame_controles,
    text="Frequência Máxima de Corte (Hz):"
)
lbl_freq_max.grid(row=4, column=0, sticky="w", pady=8)

entry_freq_max = ttk.Entry(frame_controles)
entry_freq_max.grid(row=4, column=1, sticky="ew", pady=8)
entry_freq_max.insert(0, "10.0")

# Frequência de amostragem
lbl_freq_samp = ttk.Label(
    frame_controles,
    text="Frequência de Amostragem (Hz):"
)
lbl_freq_samp.grid(row=5, column=0, sticky="w", pady=8)

entry_freq_samp = ttk.Entry(frame_controles)
entry_freq_samp.grid(row=5, column=1, sticky="ew", pady=8)
entry_freq_samp.insert(0, "100.0")

# Ordem do filtro
lbl_ordem = ttk.Label(
    frame_controles,
    text="Ordem do Filtro:"
)
lbl_ordem.grid(row=6, column=0, sticky="w", pady=8)

entry_ordem = ttk.Entry(frame_controles)
entry_ordem.grid(row=6, column=1, sticky="ew", pady=8)
entry_ordem.insert(0, "2")

# Nível de ruído
lbl_ruido = ttk.Label(
    frame_controles,
    text="Nível de Ruído:"
)
lbl_ruido.grid(row=7, column=0, sticky="w", pady=8)

entry_ruido = ttk.Entry(frame_controles)
entry_ruido.grid(row=7, column=1, sticky="ew", pady=8)
entry_ruido.insert(0, "0.1")

# Alinhamento dos botões
frame_botoes = ttk.Frame(frame_esq)
frame_botoes.pack(fill="x", padx=15, pady=10)

# Botão Gerar Sinal
btn_gerar = ttk.Button(
    frame_botoes,
    text="Gerar Sinal",
    command=gerar_sinal
)
btn_gerar.pack(side="left", expand=True, padx=2)

# Botão Aplicar Filtro
btn_aplicar = ttk.Button(
    frame_botoes,
    text="Aplicar Filtro",
    command=aplicar_filtro
)
btn_aplicar.pack(side="left", expand=True, padx=2)

# Botão Limpar
btn_limpar = ttk.Button(
    frame_botoes,
    text="Limpar",
    command=limpar_dados
)
btn_limpar.pack(side="left", expand=True, padx=2)

# --- Seção de Gráficos ---
frame_dir = ttk.LabelFrame(
    main_pane,
    text="Visualização Gráfica",
    padding=10
)
main_pane.add(frame_dir, weight=3)

# Figura dos 4 gráficos
figura, eixos = plt.subplots(2, 2, figsize=(8.5, 5.5))
eixos = eixos.flatten()

# Conversão
canvas = FigureCanvasTkAgg(figura, master=frame_dir)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Ajuste do layout inicial
figura.tight_layout(pad=2.0)
canvas.draw()

# Ajusta os campos visíveis de acordo com o filtro inicial
atualizar_campos_interface()

root.mainloop()
