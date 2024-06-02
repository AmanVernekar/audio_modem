    # for i in range(2):
    #     for j in range(5):
    #         # Generate random data
    #         # index = 10*(i*5 + j) + 9
    #         index = i*5 + j + 5
    #         _data = data[index]
    #         x = _data.real
    #         y = _data.imag
    #         # _colors = colors[index*num_data_bins:(index+1)*num_data_bins]
    #         _source_mod_seq = source_mod_seq[index*num_data_bins:(index+1)*num_data_bins]
    #         # Plot on the corresponding subplot
    #         # ax = axes[i, j]
    #         # # ax.scatter(x[_colors==mask_col], y[_colors==mask_col], c = _colors[_colors==mask_col])
    #         # ax.scatter(x, y, c = _colors)
    #         # ax.axvline(0)
    #         # ax.axhline(0)
    #         # ax.set_xlim((-50,50))
    #         # ax.set_ylim((-50,50))
    #         # ax.set_aspect('equal')

    #         errors = 0
    #         for polka, val in enumerate(_data):
    #             sent = _source_mod_seq[polka]
    #             if val.real/sent.real < 0 or val.imag/sent.imag < 0:
    #                 errors = errors + 1
    #         total_error = total_error + errors
    total_errors[g] = total_error*10/len(_data)

            # ax.set_title(f'OFDM Symbol {index + 1}\nerror % = {round(errors*100/len(_data), 2)}')
            # ax.text(10, 10, f"error % = {errors/len(_data)}")

    # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()